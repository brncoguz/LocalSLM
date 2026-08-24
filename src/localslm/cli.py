"""Command line interface.

`doctor` exists because every number this project produces is conditional on
machine state. Printing that state first — and refusing to pretend a run taken
under memory pressure is comparable to one that wasn't — is the whole ethic.
"""

from __future__ import annotations

import asyncio

import typer
from rich.console import Console
from rich.table import Table

from localslm.config import load_config
from localslm.db import connect, query, write
from localslm.host import disk_free_bytes, host_info, power_state
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
            f"{ok(power.on_ac)} {'AC' if power.on_ac else 'battery'}"
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


@app.command()
def repair(
    apply: bool = typer.Option(
        False, "--apply", help="Write the changes (default: dry run)"
    ),
) -> None:
    """Recompute the cold flag on rows recorded before residency checking.

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
        console.print("\n  [green]Nothing to repair.[/green]\n")
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
def bench(preset: str = typer.Option("quick", help="Preset from config/app.yaml")) -> None:
    """Run a benchmark preset headlessly. (Suite runner lands in phase 2.)"""
    cfg = load_config().with_preset(preset)
    console.print(f"\n  preset [bold]{preset}[/bold] · config {cfg.fingerprint()}")
    console.print("  [yellow]The suite runner is phase 2 — not wired up yet.[/yellow]")
    console.print(
        "  [dim]Phase 1 ships the measurement spine: chat, models, telemetry.[/dim]\n"
    )


if __name__ == "__main__":
    app()
