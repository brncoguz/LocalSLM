import { useCallback, useEffect, useRef, useState } from "react";
import {
  AlertTriangle, CheckCircle2, ChevronRight, Loader2, Play, RotateCcw, Trash2, XCircle,
} from "lucide-react";
import {
  api, streamGet, type CellResult, type RunDetail, type RunRow, type SuiteInfo,
} from "@/lib/api";
import { Badge, Button } from "@/components/ui/primitives";
import { ScoreBars } from "@/components/charts/ScoreBars";
import { DeterminismPanel } from "@/components/charts/DeterminismPanel";
import { JudgePanel } from "@/components/charts/JudgePanel";
import { ThinkingCurve } from "@/components/charts/ThinkingCurve";
import { TtftVsTtfat } from "@/components/charts/TtftVsTtfat";
import { cn, modelColor } from "@/lib/utils";
import { isDemo } from "@/lib/demo";

type Progress = { done: number; total: number; cell: string; item: string; passed: boolean };

const THINK_LEVELS = ["false", "low", "medium", "high"];

export function Bench({ models }: { models: string[] }) {
  const [suites, setSuites] = useState<SuiteInfo[]>([]);
  const [runs, setRuns] = useState<RunRow[]>([]);
  const [pickedModels, setPickedModels] = useState<string[]>([]);
  const [pickedSuites, setPickedSuites] = useState<string[]>([]);
  const [repeats, setRepeats] = useState(1);
  const [think, setThink] = useState<string[]>(["false"]);
  const [temps, setTemps] = useState<number[]>([0]);
  const [running, setRunning] = useState<string | null>(null);
  const [progress, setProgress] = useState<Progress | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [openRun, setOpenRun] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const refresh = useCallback(async () => {
    const [s, r] = await Promise.all([api.suites(), api.runs()]);
    setSuites(s);
    setRuns(r);
    if (!pickedSuites.length) setPickedSuites(s.map((x) => x.name));
  }, [pickedSuites.length]);

  useEffect(() => { void refresh(); }, [refresh]);
  useEffect(() => { if (!pickedModels.length && models.length) setPickedModels(models); },
    [models, pickedModels.length]);

  const itemCount = suites.filter((s) => pickedSuites.includes(s.name))
    .reduce((n, s) => n + s.items, 0);
  // Mirrors BenchSpec.total_requests: warmups are per suite, not per item.
  const cellCount = pickedModels.length * Math.max(think.length, 1) * Math.max(temps.length, 1);
  const estimate = cellCount * itemCount * repeats + cellCount * pickedSuites.length;

  /** Start a run from the form, or from a spec lifted off an earlier one.
   *  Same path either way — a rerun that took a different route could drift
   *  from the original in ways the comparison would then read as a result. */
  async function start(override?: Parameters<typeof api.startRun>[0]) {
    setError(null);
    setProgress(null);
    try {
      const { run_id } = await api.startRun(
        override ?? {
          models: pickedModels, suites: pickedSuites, repeats,
          think_levels: think, temperatures: temps, label: "ui",
        },
      );
      setRunning(run_id);
      const ctrl = new AbortController();
      abortRef.current = ctrl;
      await streamGet(`/bench/runs/${run_id}/stream`, (ev, data) => {
        if (ev === "unit") {
          setProgress({
            done: data.done, total: data.total, cell: data.cell,
            item: `${data.suite}/${data.item}`, passed: data.passed,
          });
        } else if (ev === "aborted") {
          setError(`Run stopped: ${data.reason}`);
        }
      }, ctrl.signal);
    } catch (e: any) {
      const d = e?.detail;
      setError(
        d?.error === "engine_busy" ? "Another run is already using the engine."
        : d?.error === "model_missing" ? `Not installed: ${d.models?.join(", ")}`
        : d?.error === "memory_pressure" ? d.message
        : (e?.message ?? "Could not start the run."),
      );
    } finally {
      setRunning(null);
      setProgress(null);
      void refresh();
    }
  }

  /** Repeat an earlier run's experiment exactly.
   *
   *  The plan is re-read from the server rather than reconstructed from the
   *  row, because the row carries only what the list needs to render — think
   *  levels, repeats and temperatures live in the stored plan, and silently
   *  defaulting any of them would produce a run that looks comparable and
   *  isn't. */
  async function rerun(runId: string) {
    setError(null);
    try {
      const detail = await api.run(runId);
      const plan = detail.plan;
      if (!plan?.models?.length || !plan?.suites?.length) {
        setError("That run did not record enough of its plan to repeat it.");
        return;
      }
      await start({
        models: plan.models,
        suites: plan.suites.map((s) => s.name),
        repeats: plan.repeats,
        think_levels: plan.think_levels,
        // Runs from before the plan recorded temperatures fall back to the
        // default rather than inventing a sweep the original never had.
        temperatures: plan.temperatures ?? [0],
        label: `rerun of ${detail.label || runId.slice(0, 8)}`,
      });
    } catch (e: any) {
      setError(e?.detail?.message ?? e?.message ?? "Could not repeat that run.");
    }
  }

  const toggle = (list: string[], set: (v: string[]) => void, value: string) =>
    set(list.includes(value) ? list.filter((x) => x !== value) : [...list, value]);

  return (
    <div className="h-full overflow-y-auto px-5 py-5">
      <div className="mx-auto max-w-5xl space-y-5">
        <header>
          <h2 className="text-base font-semibold">Benchmarks</h2>
          <p className="mt-0.5 text-[12px] text-faint">
            Every model answers the same items, so results are compared pairwise.
          </p>
        </header>

        <div className="rounded-xl border border-border bg-surface p-4">
          <div className="grid gap-4 sm:grid-cols-2">
            <Picker title="Models" options={models} picked={pickedModels}
              onToggle={(v) => toggle(pickedModels, setPickedModels, v)} colour />
            <Picker
              title="Suites"
              options={suites.map((s) => s.name)}
              picked={pickedSuites}
              onToggle={(v) => toggle(pickedSuites, setPickedSuites, v)}
              meta={Object.fromEntries(suites.map((s) => [s.name, `${s.items}`]))}
            />
          </div>

          <div className="mt-4 flex flex-wrap items-end gap-4 border-t border-border pt-3.5">
            <label className="text-[11px] text-faint">
              <span className="mb-1 block">repeats</span>
              <input type="number" min={1} max={10} value={repeats}
                onChange={(e) => setRepeats(Math.max(1, Number(e.target.value)))}
                className="tnum w-16 rounded-md border border-border bg-bg px-2 py-1 text-[12px] text-fg focus:outline-none focus:ring-1 focus:ring-accent/50" />
            </label>

            <div className="text-[11px] text-faint">
              <span className="mb-1 block">thinking</span>
              <div className="flex overflow-hidden rounded-md border border-border">
                {THINK_LEVELS.map((lvl) => (
                  <button key={lvl} onClick={() => toggle(think, setThink, lvl)}
                    className={cn("px-2 py-1 text-[11px] transition-colors",
                      think.includes(lvl) ? "bg-warn/15 text-warn" : "text-faint hover:text-muted")}>
                    {lvl === "false" ? "off" : lvl}
                  </button>
                ))}
              </div>
            </div>

            <div className="text-[11px] text-faint">
              <span className="mb-1 block">temperature</span>
              <div className="flex overflow-hidden rounded-md border border-border">
                {[0, 0.7, 1].map((t) => (
                  <button key={t}
                    onClick={() => setTemps(
                      temps.includes(t) ? temps.filter((x) => x !== t) : [...temps, t].sort(),
                    )}
                    className={cn("tnum px-2 py-1 text-[11px] transition-colors",
                      temps.includes(t) ? "bg-accent/15 text-accent" : "text-faint hover:text-muted")}>
                    {t}
                  </button>
                ))}
              </div>
            </div>

            <div className="ml-auto flex items-center gap-3">
              <span className="tnum text-[11px] text-faint">
                ~{estimate.toLocaleString()} requests
              </span>
              <Button onClick={() => void start()}
                title={isDemo ? "Needs a local Ollama — this is a recorded demo" : undefined}
                disabled={isDemo || !!running || !pickedModels.length || !pickedSuites.length || !think.length || !temps.length}>
                {running ? <Loader2 size={13} className="animate-spin" /> : <Play size={13} />}
                {running ? "Running" : "Run"}
              </Button>
              {running && (
                <Button variant="outline" size="sm"
                  onClick={() => { void api.abortRun(running); abortRef.current?.abort(); }}>
                  Stop
                </Button>
              )}
            </div>
          </div>

          {progress && (
            <div className="mt-3.5 border-t border-border pt-3">
              <div className="h-1 overflow-hidden rounded-full bg-surface-2">
                <div className="h-full bg-accent transition-[width] duration-200"
                  style={{ width: `${(progress.done / Math.max(progress.total, 1)) * 100}%` }} />
              </div>
              <div className="tnum mt-1.5 flex items-center gap-2 text-[11px] text-faint">
                <span className="text-muted">{progress.done}/{progress.total}</span>
                <span style={{ color: modelColor(progress.cell.split(" ")[0]) }}>{progress.cell}</span>
                <span>{progress.item}</span>
                {progress.passed
                  ? <CheckCircle2 size={11} className="text-good" />
                  : <XCircle size={11} className="text-bad" />}
              </div>
            </div>
          )}

          {error && (
            <p className="mt-3 rounded-md bg-bad/8 px-2.5 py-1.5 text-[11px] text-bad">{error}</p>
          )}
        </div>

        <section>
          <h3 className="mb-2 text-sm font-semibold">Runs</h3>
          {runs.length === 0 && (
            <p className="rounded-lg border border-dashed border-border px-3 py-6 text-center text-[11px] text-faint">
              No runs yet.
            </p>
          )}
          <div className="space-y-2">
            {runs.map((r) => (
              <RunCard key={r.id} run={r}
                open={openRun === r.id}
                busy={!!running}
                onToggle={() => setOpenRun(openRun === r.id ? null : r.id)}
                onRerun={() => void rerun(r.id)}
                onDeleted={() => { setOpenRun(null); void refresh(); }} />
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}

function Picker({
  title, options, picked, onToggle, meta, colour,
}: {
  title: string; options: string[]; picked: string[];
  onToggle: (v: string) => void; meta?: Record<string, string>; colour?: boolean;
}) {
  return (
    <div>
      <p className="mb-1.5 text-[11px] uppercase tracking-wide text-faint">{title}</p>
      <div className="flex flex-wrap gap-1.5">
        {options.map((o) => {
          const on = picked.includes(o);
          return (
            <button key={o} onClick={() => onToggle(o)}
              className={cn("flex items-center gap-1.5 rounded-md border px-2 py-1 text-[11px] transition-colors",
                on ? "border-accent/40 bg-accent/10 text-fg" : "border-border text-faint hover:text-muted")}>
              {colour && (
                <span className="h-1.5 w-1.5 rounded-full"
                  style={{ background: on ? modelColor(o) : "var(--color-faint)" }} />
              )}
              {o}
              {meta?.[o] && <span className="tnum text-faint">{meta[o]}</span>}
            </button>
          );
        })}
      </div>
    </div>
  );
}

function RunCard({
  run, open, busy: engineBusy, onToggle, onRerun, onDeleted,
}: {
  run: RunRow; open: boolean; busy: boolean;
  onToggle: () => void; onRerun: () => void; onDeleted: () => void;
}) {
  const [detail, setDetail] = useState<RunDetail | null>(null);
  const [confirming, setConfirming] = useState(false);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (open && !detail) void api.run(run.id).then(setDetail);
  }, [open, detail, run.id]);

  async function remove() {
    setBusy(true);
    setErr(null);
    try {
      await api.deleteRun(run.id);
      onDeleted();
    } catch (e: any) {
      setErr(e?.detail?.message ?? "Could not delete this run.");
      setBusy(false);
      setConfirming(false);
    }
  }

  const tone = run.status === "done" ? "good" : run.status === "running" ? "accent" : "bad";
  return (
    <div className="rounded-lg border border-border bg-surface">
      <div className="flex items-center gap-1 pr-2">
        <button onClick={onToggle} className="flex min-w-0 flex-1 items-center gap-3 px-3 py-2.5 text-left">
          <ChevronRight size={13} className={cn("shrink-0 text-faint transition-transform", open && "rotate-90")} />
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-[12px] font-medium">{run.suite}</span>
              <Badge tone={tone as never}>{run.status}</Badge>
              {run.label && <Badge>{run.label}</Badge>}
            </div>
            <div className="tnum mt-0.5 flex flex-wrap gap-2 text-[11px] text-faint">
              <span>{new Date(run.ts * 1000).toLocaleString()}</span>
              <span>{run.done_units}/{run.planned_units} items</span>
              {run.elapsed_s != null && <span>{run.elapsed_s.toFixed(0)}s</span>}
              <span>{run.models.length} models</span>
            </div>
          </div>
          {run.best_score != null && (
            <span className="tnum shrink-0 text-sm font-semibold">{run.best_score.toFixed(2)}</span>
          )}
        </button>
        <Button
          size="icon"
          variant="ghost"
          title={
            engineBusy
              ? "A run is already using the engine"
              : `Run this experiment again — same models, suites, think levels and repeats`
          }
          disabled={busy || engineBusy}
          onClick={onRerun}
        >
          <RotateCcw size={13} />
        </Button>
        <Button size="icon" variant="ghost" title="Delete this run"
          disabled={busy} onClick={() => setConfirming(true)}>
          {busy ? <Loader2 size={13} className="animate-spin" /> : <Trash2 size={13} />}
        </Button>
      </div>

      {confirming && (
        <div className="flex flex-wrap items-center gap-2 border-t border-bad/25 bg-bad/8 px-3 py-2 text-[11px]">
          <AlertTriangle size={13} className="shrink-0 text-bad" />
          <span className="text-muted">
            {/* A run *is* its measurements, unlike a chat session where the
                transcript is disposable and the measurement is the artifact. */}
            Delete this run and its {run.done_units} measurements? They disappear
            from the Stats charts too.
          </span>
          <div className="ml-auto flex gap-1">
            <Button size="sm" variant="ghost" onClick={() => setConfirming(false)}>Cancel</Button>
            <Button size="sm" variant="danger" onClick={() => void remove()}>Delete</Button>
          </div>
        </div>
      )}

      {err && <p className="border-t border-border px-3 py-1.5 text-[11px] text-bad">{err}</p>}

      {open && (
        <div className="border-t border-border p-3">
          {detail ? (
            <div className="space-y-4">
              <RunCells cells={detail.cells} />
              <ThinkingCurve runId={run.id} />
              <TtftVsTtfat runId={run.id} />
              <DeterminismPanel runId={run.id} />
              <JudgePanel runId={run.id} />
            </div>
          ) : (
            <p className="text-[11px] text-faint">Loading…</p>
          )}
        </div>
      )}
    </div>
  );
}

function RunCells({ cells }: { cells: CellResult[] }) {
  if (!cells.length) return <p className="text-[11px] text-faint">No scored cells.</p>;
  const tasks = [...new Set(cells.flatMap((c) => Object.keys(c.tasks)))].sort();
  return (
    <div className="space-y-4">
      <ScoreBars cells={cells} />
      <div className="overflow-x-auto">
        <table className="tnum w-full min-w-[520px] text-[11px]">
          <thead>
            <tr className="text-left text-faint">
              <th className="pb-2 font-medium">cell</th>
              <th className="pb-2 text-right font-medium">n</th>
              <th className="pb-2 text-right font-medium">score</th>
              <th className="pb-2 text-right font-medium">95% CI</th>
              {tasks.map((t) => <th key={t} className="pb-2 text-right font-medium">{t}</th>)}
            </tr>
          </thead>
          <tbody>
            {cells.map((c) => (
              <tr key={c.cell_key} className="border-t border-border">
                <td className="py-1.5" style={{ color: modelColor(c.model) }}>{c.label}</td>
                <td className="py-1.5 text-right text-faint">{c.n}</td>
                <td className="py-1.5 text-right font-medium text-muted">{c.score.toFixed(2)}</td>
                <td className="py-1.5 text-right text-faint">
                  [{c.ci_lo.toFixed(2)}, {c.ci_hi.toFixed(2)}]
                  {/* Flagged when this cell's interval overlaps the one above:
                      the ordering here is not evidence of a real difference. */}
                  {c.tied_with_previous && <span className="ml-1 text-warn" title="Overlaps the row above — not a significant difference">≈</span>}
                </td>
                {tasks.map((t) => (
                  <td key={t} className="py-1.5 text-right text-muted">
                    {c.tasks[t] ? c.tasks[t].accuracy.toFixed(2) : <span className="text-faint">—</span>}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-[10px] text-faint">
        <span className="text-warn">≈</span> marks a score whose interval overlaps the row above,
        meaning the data cannot separate them.
      </p>
    </div>
  );
}
