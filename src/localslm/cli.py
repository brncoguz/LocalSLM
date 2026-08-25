"""Command line interface.

`doctor` exists because every number this project produces is conditional on
machine state. Printing that state first — and refusing to pretend a run taken
under memory pressure is comparable to one that wasn't — is the whole ethic.
"""

from __future__ import annotations

import asyncio
import json

import typer
from rich.console import Console
from rich.table import Table

from localslm.config import load_config
from localslm.db import connect, query, write
from localslm.host import disk_free_bytes, host_info, power_label, power_state
from localslm.runtimes.base import COLD_LOAD_MS_FALLBACK
from localslm.runtimes.ollama import OllamaRuntime
from localslm.telemetry.sampler import SystemSampler

app = typer.Typer(
    add_completion=False, help="LocalSLM — measure models on your own machine."
)
console = Console()


def _runtime() -> OllamaRuntime:
    cfg = load_config()
    return OllamaRuntime(cfg["ollama.host"])


@app.command()
def doctor() -> None:
    """Check the host: Ollama, disk, memory, power."""

    async def main() -> None:
        host = host_info()
        rt = _runtime()
        health = await rt.health()
        power = power_state()
        sampler = SystemSampler(interval_s=0.2)
        sampler.start()
        await asyncio.sleep(0.6)
        sample = sampler.latest
        sampler.stop()

        t = Table(show_header=False, box=None, padding=(0, 2))
        t.add_column(style="dim")
        t.add_column()

        def ok(flag: bool) -> str:
            return "[green]✓[/green]" if flag else "[red]✗[/red]"

        t.add_row("machine", f"{host.chip} · {host.cores} cores · {host.ram_gb:.0f} GB")
        t.add_row("macOS", host.macos)
        t.add_row("python", host.python)
        t.add_row(
            "ollama",
            f"{ok(health['ok'])} {health.get('version') or health.get('error')} @ {health['endpoint']}",
        )
        free_gb = disk_free_bytes() / 1e9
        t.add_row("disk free", f"{ok(free_gb > 10)} {free_gb:.0f} GB")
        if sample:
            pressure = sample.pressure
            t.add_row(
                "memory",
                f"{ok(pressure == 'normal')} {sample.free_bytes / 1e9:.1f} GB free · "
                f"swap {sample.swap_used_bytes / 1e9:.1f} GB · pressure {pressure}",
            )
            t.add_row("ollama rss", f"{sample.ollama_rss_bytes / 1e9:.2f} GB")
        t.add_row(
            "power",
            f"{ok(bool(power.on_ac))} {power_label(power)}"
            + (f" · {power.battery_pct}%" if power.battery_pct is not None else "")
            + ("" if power.on_ac else "  [yellow]battery throttles Apple Silicon[/yellow]"),
        )
        t.add_row("env hash", host.hash())

        console.print()
        console.print(t)
        if health["ok"]:
            models = await rt.list_models()
            console.print(f"\n  [dim]{len(models)} models installed[/dim]")
        console.print()
        await rt.aclose()

    asyncio.run(main())


@app.command()
def models() -> None:
    """List installed models with their measured headline numbers."""

    async def main() -> None:
        rt = _runtime()
        installed = await rt.list_models()
        loaded = {m.name for m in await rt.loaded()}
        conn = connect()
        stats = {
            r["model"]: r
            for r in query(
                conn,
                "SELECT model, COUNT(*) n, AVG(decode_tps) tps, AVG(ttfat_ms) ttfat "
                "FROM requests WHERE ok=1 AND excluded=0 AND decode_tps IS NOT NULL GROUP BY model",
            )
        }
        t = Table(title="Installed models", title_justify="left")
        t.add_column("model")
        t.add_column("quant")
        t.add_column("params", justify="right")
        t.add_column("size", justify="right")
        t.add_column("state")
        t.add_column("tok/s", justify="right")
        t.add_column("TTFAT", justify="right")
        t.add_column("n", justify="right")
        for m in sorted(installed, key=lambda x: x.size_bytes):
            s = stats.get(m.ref.name)
            t.add_row(
                m.ref.name,
                m.ref.quant or "-",
                f"{m.ref.params_b:.1f}B" if m.ref.params_b else "-",
                f"{m.size_bytes / 1e9:.1f} GB",
                "[green]warm[/green]" if m.ref.name in loaded else "[dim]cold[/dim]",
                f"{s['tps']:.1f}" if s and s["tps"] else "[dim]—[/dim]",
                f"{s['ttfat']:.0f} ms" if s and s["ttfat"] else "[dim]—[/dim]",
                str(s["n"]) if s else "[dim]0[/dim]",
            )
        console.print()
        console.print(t)
        console.print()
        await rt.aclose()

    asyncio.run(main())


@app.command("export-demo")
def export_demo(
    out: str = typer.Option("ui/src/fixtures/demo.json", help="Where to write the fixture"),
) -> None:
    """Snapshot the API into a fixture the static demo can serve.

    Nobody reviewing this repository is going to install Ollama and pull several
    gigabytes of weights, so the published demo has to run on recorded data.

    The snapshot is taken by calling the real application through a test client
    rather than by re-querying the database. Any hand-written export would drift
    from what the API actually serves the moment a route changed, and a demo
    that disagrees with the app is worse than no demo.
    """
    from fastapi.testclient import TestClient

    from localslm.api.main import app as fastapi_app
    from localslm.config import path as repo_path

    target = repo_path(*out.split("/"))
    target.parent.mkdir(parents=True, exist_ok=True)

    snapshot: dict[str, object] = {}
    with TestClient(fastapi_app) as client:

        def capture(path: str) -> object | None:
            r = client.get(path)
            if r.status_code != 200:
                console.print(f"  [yellow]skipped {path} ({r.status_code})[/yellow]")
                return None
            snapshot[path] = r.json()
            return r.json()

        for path in (
            "/api/health",
            "/api/models",
            "/api/suites",
            "/api/stats/summary",
            "/api/stats/timeline?limit=1000",
            "/api/stats/disagreements?limit=40",
            "/api/runs?limit=50",
        ):
            capture(path)

        runs = snapshot.get("/api/runs?limit=50") or []
        # Every completed run, so the demo's run list expands like the real one.
        for run in [r for r in runs if r["status"] in ("done", "aborted")][:8]:
            rid = run["id"]
            for suffix in ("", "/thinking", "/determinism"):
                capture(f"/api/runs/{rid}{suffix}")

        for model in (snapshot.get("/api/models") or {}).get("installed", []):
            capture(f"/api/models/{model['name']}/history?limit=100")
            capture(f"/api/models/{model['name']}/detail")

    target.write_text(json.dumps(snapshot, indent=1, sort_keys=True))
    size = target.stat().st_size
    console.print(
        f"\n  wrote [bold]{len(snapshot)}[/bold] endpoints to {out} "
        f"({size / 1024:.0f} KB)\n"
    )


def _stale_cells(conn) -> list[tuple]:
    """Frozen cell aggregates that disagree with the rows they summarise.

    Runs benchmarked before the freeze respected `requests.excluded` counted
    swapped measurements into their score, while every endpoint reading the
    same run excluded them. The `results` rows are still there, so the correct
    aggregate can be recomputed rather than guessed.
    """
    from localslm.bench.stats import bootstrap_ci

    stale = []
    for cell in query(
        conn, "SELECT run_id, cell_key, model, n, n_excluded, score_mean FROM cells"
    ):
        graded = query(
            conn,
            """SELECT res.score, res.passed, req.excluded
               FROM results res JOIN requests req ON req.id = res.request_id
               WHERE res.run_id = ? AND res.cell_key = ?""",
            (cell["run_id"], cell["cell_key"]),
        )
        kept = [g for g in graded if not g["excluded"]]
        n_excluded = len(graded) - len(kept)
        if cell["n"] == len(kept) and (cell["n_excluded"] or 0) == n_excluded:
            continue
        scores = [g["score"] or 0.0 for g in kept]
        ci = bootstrap_ci(scores) if scores else None
        stale.append(
            (
                cell,
                {
                    "n": len(kept),
                    "n_excluded": n_excluded,
                    "n_failed": sum(1 for g in kept if not g["passed"]),
                    "score_mean": ci.mean if ci else None,
                    "score_ci_lo": ci.lo if ci else None,
                    "score_ci_hi": ci.hi if ci else None,
                    "pass_rate": (
                        sum(1 for g in kept if g["passed"]) / len(kept) if kept else None
                    ),
                },
            )
        )
    return stale


def _repair_cells(conn, apply: bool) -> None:
    stale = _stale_cells(conn)
    if not stale:
        console.print("  [green]Frozen cell aggregates agree with their rows.[/green]")
        return

    t = Table(
        title=f"{len(stale)} frozen aggregates include excluded measurements",
        title_justify="left",
    )
    t.add_column("model")
    t.add_column("n", justify="right")
    t.add_column("score", justify="right")
    for cell, fixed in stale:
        old_score = "—" if cell["score_mean"] is None else f"{cell['score_mean']:.3f}"
        new_score = "—" if fixed["score_mean"] is None else f"{fixed['score_mean']:.3f}"
        t.add_row(
            cell["model"],
            f"{cell['n']} → [bold]{fixed['n']}[/bold]",
            f"{old_score} → [bold]{new_score}[/bold]",
        )
    console.print()
    console.print(t)

    if not apply:
        return

    with write(conn):
        for cell, fixed in stale:
            conn.execute(
                "UPDATE cells SET n = ?, n_excluded = ?, n_failed = ?, score_mean = ?, "
                "score_ci_lo = ?, score_ci_hi = ?, pass_rate = ? "
                "WHERE run_id = ? AND cell_key = ?",
                (
                    fixed["n"],
                    fixed["n_excluded"],
                    fixed["n_failed"],
                    fixed["score_mean"],
                    fixed["score_ci_lo"],
                    fixed["score_ci_hi"],
                    fixed["pass_rate"],
                    cell["run_id"],
                    cell["cell_key"],
                ),
            )
    console.print(f"  [green]Refroze {len(stale)} cell aggregates.[/green]")


@app.command()
def repair(
    apply: bool = typer.Option(
        False, "--apply", help="Write the changes (default: dry run)"
    ),
) -> None:
    """Re-score history when a metric definition changes.

    Two repairs, both dry-run by default:

    Cold flag on rows recorded before residency checking.

    Early rows used a 100 ms load-duration threshold. Ollama reports ~150-300 ms
    of handle setup even when weights are resident, so those rows are all
    flagged cold. This re-derives the flag with the documented fallback rule
    (`load_ms > COLD_LOAD_MS_FALLBACK`), which is what we use today whenever
    residency is unknown — so history ends up scored by the same rule as new data.
    """
    conn = connect()
    rows = query(
        conn,
        "SELECT id, model, srv_load_ms, cold FROM requests "
        "WHERE srv_load_ms IS NOT NULL ORDER BY ts",
    )
    changes = [
        (
            r["id"],
            r["model"],
            r["srv_load_ms"],
            r["cold"],
            int(r["srv_load_ms"] > COLD_LOAD_MS_FALLBACK),
        )
        for r in rows
        if r["cold"] != int(r["srv_load_ms"] > COLD_LOAD_MS_FALLBACK)
    ]

    if not changes:
        console.print("\n  [green]Cold flags agree with the current rule.[/green]")
        _repair_cells(conn, apply)
        console.print()
        return

    t = Table(
        title=f"{len(changes)} rows disagree with the current rule", title_justify="left"
    )
    t.add_column("model")
    t.add_column("load", justify="right")
    t.add_column("cold", justify="center")
    for _id, model, load, old, new in changes[:25]:
        t.add_row(model, f"{load:.0f} ms", f"{old} → [bold]{new}[/bold]")
    console.print()
    console.print(t)
    if len(changes) > 25:
        console.print(f"  [dim]…and {len(changes) - 25} more[/dim]")

    _repair_cells(conn, apply)

    if not apply:
        console.print("\n  [yellow]Dry run.[/yellow] Re-run with --apply to write.\n")
        return

    with write(conn):
        conn.executemany(
            "UPDATE requests SET cold = ? WHERE id = ?",
            [(new, _id) for _id, _, _, _, new in changes],
        )
    console.print(f"\n  [green]Updated {len(changes)} rows.[/green]\n")


@app.command()
def suites() -> None:
    """List available test suites."""
    from localslm.bench.suites import available, load

    t = Table(title="Test suites", title_justify="left")
    t.add_column("suite")
    t.add_column("items", justify="right")
    t.add_column("verified", justify="right")
    t.add_column("grader")
    t.add_column("hash")
    for name in available():
        s = load(name)
        t.add_row(
            name,
            str(len(s)),
            f"{s.verified}" if s.verified == len(s) else f"[yellow]{s.verified}[/yellow]",
            ", ".join(sorted(s.tasks)),
            s.content_hash(),
        )
    console.print()
    console.print(t)
    console.print()


@app.command()
def bench(
    models: str = typer.Option(
        "", help="Comma-separated; defaults to every installed model"
    ),
    suite: str = typer.Option("", help="Comma-separated; defaults to all suites"),
    repeats: int = typer.Option(1, help="Measured passes per item"),
    think: str = typer.Option("false", help="Comma-separated: false,low,medium,high"),
    label: str = typer.Option("cli", help="Label recorded on the run"),
) -> None:
    """Run a benchmark and print per-cell scores with confidence intervals."""
    import asyncio

    from localslm.api.deps import init_state, shutdown_state
    from localslm.bench.runner import BenchRunner
    from localslm.bench.spec import BenchSpec
    from localslm.bench.suites import available, load
    from localslm.db import new_id

    async def main() -> None:
        st = init_state()
        try:
            installed = await st.runtime.list_models()
            chosen = [m.strip() for m in models.split(",") if m.strip()] or [
                m.ref.name for m in installed
            ]
            names = {m.ref.name for m in installed}
            missing = [m for m in chosen if m not in names]
            if missing:
                console.print(f"  [red]not installed:[/red] {', '.join(missing)}")
                return

            capable = {}
            for name in chosen:
                info = await st.runtime.show(name)
                capable[name] = info.capabilities.thinking

            suite_names = [x.strip() for x in suite.split(",") if x.strip()] or available()
            levels: list[bool | str] = []
            for raw in think.split(","):
                raw = raw.strip().lower()
                levels.append(False if raw in ("false", "off", "") else raw)

            spec = BenchSpec(
                models=chosen,
                suites=[load(n) for n in suite_names],
                think_levels=levels,
                repeats=repeats,
                warmups=1,
                thinking_capable=capable,
                power_source=power_label(),
            )
            total = spec.total_requests()
            console.print(
                f"\n  {len(spec.cells())} cells x {sum(len(s) for s in spec.suites)} items "
                f"x {repeats} reps = [bold]{total}[/bold] requests"
            )

            run_id = new_id()
            await st.lease.acquire_exclusive("bench", run_id)
            cells: list = []
            try:
                runner = BenchRunner(st, spec, run_id, label)
                with console.status("running...") as status:
                    async for ev in runner.run():
                        if ev.kind == "unit":
                            d = ev.data
                            status.update(
                                f"[{d['done']}/{d['total']}] {d['cell']} - "
                                f"{d['suite']}/{d['item']} {'ok' if d['passed'] else 'FAIL'}"
                            )
                        elif ev.kind == "warning":
                            console.print(f"  [yellow]{ev.data['message']}[/yellow]")
                        elif ev.kind == "aborted":
                            console.print(f"  [red]aborted: {ev.data['reason']}[/red]")
                        elif ev.kind == "done":
                            cells = ev.data["cells"]
                            console.print(
                                f"  finished in {ev.data['elapsed_s']}s - run {run_id}"
                            )
            finally:
                await st.lease.release_exclusive()

            if cells:
                _print_cells(cells)
        finally:
            await shutdown_state()

    asyncio.run(main())


def _print_cells(cells: list) -> None:
    """Score per cell, with the interval beside it.

    The interval is not decoration: at these sample sizes most apparent gaps
    are noise, and printing a bare mean invites reading a ranking into it.
    """
    tasks = sorted({t for c in cells for t in c["tasks"]})
    t = Table(title="Results", title_justify="left")
    t.add_column("model")
    t.add_column("settings")
    t.add_column("n", justify="right")
    t.add_column("score", justify="right")
    t.add_column("95% CI", justify="right")
    for task in tasks:
        t.add_column(task, justify="right")

    best = max((c["score"] for c in cells), default=0)
    for c in sorted(cells, key=lambda x: -x["score"]):
        axes = ", ".join(f"{k}={v}" for k, v in c["axes"].items()) or "-"
        score = f"{c['score']:.2f}"
        row = [
            c["model"],
            axes,
            str(c["n"]),
            f"[bold]{score}[/bold]" if c["score"] == best else score,
            f"[dim][{c['ci_lo']:.2f}, {c['ci_hi']:.2f}][/dim]",
        ]
        for task in tasks:
            m = c["tasks"].get(task)
            row.append(f"{m['accuracy']:.2f}" if m else "[dim]-[/dim]")
        t.add_row(*row)
    console.print()
    console.print(t)
    console.print()


if __name__ == "__main__":
    app()
