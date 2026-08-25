import { useCallback, useEffect, useRef, useState } from "react";
import {
  AlertTriangle, ChevronRight, Loader2, Play, RotateCcw, Trash2, X,
} from "lucide-react";
import {
  api, streamGet, type CellResult, type ModelCard, type RunDetail, type RunRow,
  type SuiteInfo,
} from "@/lib/api";

/** The two spellings the think axis uses for "no reasoning". Mirrors the
 *  runner, which maps both onto `False`. */
const isOff = (t: string) => t === "false" || t === "off";
import { Badge, Button } from "@/components/ui/primitives";
import { BenchLive } from "@/components/BenchLive";
import { ItemBrowser, type LiveUnit } from "@/components/ItemBrowser";
import { ScoreBars } from "@/components/charts/ScoreBars";
import { DeterminismPanel } from "@/components/charts/DeterminismPanel";
import { JudgePanel } from "@/components/charts/JudgePanel";
import { ThinkingCurve } from "@/components/charts/ThinkingCurve";
import { TtftVsTtfat } from "@/components/charts/TtftVsTtfat";
import { cn, modelColor } from "@/lib/utils";
import { isDemo } from "@/lib/demo";



const THINK_LEVELS = ["false", "low", "medium", "high"];

export function Bench({ models: installed }: { models: ModelCard[] }) {
  const models = installed.map((m) => m.name);
  const canThink = new Map(installed.map((m) => [m.name, m.thinking]));
  const [suites, setSuites] = useState<SuiteInfo[]>([]);
  const [runs, setRuns] = useState<RunRow[]>([]);
  const [pickedModels, setPickedModels] = useState<string[]>([]);
  const [pickedSuites, setPickedSuites] = useState<string[]>([]);
  const [repeats, setRepeats] = useState(1);
  const [think, setThink] = useState<string[]>(["false"]);
  const [temps, setTemps] = useState<number[]>([0]);
  const [running, setRunning] = useState<string | null>(null);
  // Every unit seen this run, kept so the scoreboard and the heartbeat strip
  // can be derived rather than accumulated in three separate pieces of state.
  const [units, setUnits] = useState<LiveUnit[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [openRun, setOpenRun] = useState<string | null>(null);
  // Set by clicking a suite or label tag on a run card. With a few dozen runs
  // the list is mostly scrolling, and the tag you want to filter by is already
  // on screen and under the cursor.
  const [tagFilter, setTagFilter] = useState<{ kind: "suite" | "label"; value: string } | null>(null);
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

  // Mirrors BenchSpec.cells(): a model that cannot think has its non-off think
  // levels dropped, because running four of them would produce four identical
  // rows. Multiplying blindly by think.length promised 84 requests for a run
  // that was only ever going to make 20.
  const cellsFor = (name: string) => {
    const levels = canThink.get(name) === false ? think.filter(isOff) : think;
    return Math.max(levels.length, 0) * Math.max(temps.length, 1);
  };
  const cellCount = pickedModels.reduce((n, m) => n + cellsFor(m), 0);
  // Mirrors BenchSpec.total_requests: warmups are per suite, not per item.
  const estimate = cellCount * itemCount * repeats + cellCount * pickedSuites.length;

  // Models the think axis does not apply to, split by how badly. Losing the
  // levels is a surprise; contributing no cells at all is a silent no-op.
  const wantsThinking = think.some((t) => !isOff(t));
  const nonThinking = pickedModels.filter((m) => canThink.get(m) === false);
  const thinkIgnored = wantsThinking ? nonThinking : [];
  const skippedEntirely = think.every((t) => !isOff(t)) ? nonThinking : [];

  /** Start a run from the form, or from a spec lifted off an earlier one.
   *  Same path either way — a rerun that took a different route could drift
   *  from the original in ways the comparison would then read as a result. */
  async function start(override?: Parameters<typeof api.startRun>[0]) {
    setError(null);
    setUnits([]);
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
          setUnits((prev) => [...prev, data as LiveUnit]);
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
      // The units stay on screen after the run ends: the last thing you were
      // watching is usually the thing you want to look at again.
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

  const shownRuns = tagFilter
    ? runs.filter((r) =>
        tagFilter.kind === "label"
          ? r.label === tagFilter.value
          : r.suite.split(",").map((x) => x.trim()).includes(tagFilter.value),
      )
    : runs;

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

          {units.length > 0 && (
            <BenchLive
              units={units}
              onStop={
                running
                  ? () => {
                      void api.abortRun(running);
                      abortRef.current?.abort();
                    }
                  : undefined
              }
            />
          )}

          {/* Said before the run rather than discovered during it. The planner
              drops think levels a model cannot use — four identical rows help
              nobody — but that silently makes the run a fraction of the size
              the estimate promised. */}
          {!!skippedEntirely.length && (
            <p className="mt-3 rounded-md bg-bad/8 px-2.5 py-1.5 text-[11px] text-bad">
              {skippedEntirely.join(", ")} cannot think, and every selected think
              level needs it — {skippedEntirely.length === 1 ? "it" : "they"} will
              contribute nothing. Add <span className="font-medium">off</span>, or
              deselect {skippedEntirely.length === 1 ? "it" : "them"}.
            </p>
          )}
          {!skippedEntirely.length && !!thinkIgnored.length && (
            <p className="mt-3 rounded-md bg-warn/8 px-2.5 py-1.5 text-[11px] text-warn">
              {thinkIgnored.join(", ")} cannot think, so the think levels are
              skipped for {thinkIgnored.length === 1 ? "it" : "them"} — running
              them would repeat identical rows. Already excluded from the estimate.
            </p>
          )}

          {error && (
            <p className="mt-3 rounded-md bg-bad/8 px-2.5 py-1.5 text-[11px] text-bad">{error}</p>
          )}
        </div>

        <section>
          <div className="mb-2 flex flex-wrap items-center gap-2">
            <h3 className="text-sm font-semibold">Runs</h3>
            {tagFilter && (
              <button
                onClick={() => setTagFilter(null)}
                className="flex items-center gap-1 rounded-md bg-accent/12 px-1.5 py-0.5 text-[11px] text-accent"
                title="Clear this filter"
              >
                {tagFilter.kind}: {tagFilter.value}
                <X size={10} />
              </button>
            )}
            <span className="tnum ml-auto text-[11px] text-faint">
              {shownRuns.length}
              {shownRuns.length !== runs.length && ` of ${runs.length}`} runs
            </span>
          </div>
          {shownRuns.length === 0 && (
            <p className="rounded-lg border border-dashed border-border px-3 py-6 text-center text-[11px] text-faint">
              {runs.length ? "No runs match that tag." : "No runs yet."}
            </p>
          )}
          <div className="space-y-2">
            {shownRuns.map((r) => (
              <RunCard key={r.id} run={r}
                open={openRun === r.id}
                busy={!!running}
                onToggle={() => setOpenRun(openRun === r.id ? null : r.id)}
                onTag={(kind, value) =>
                  setTagFilter((cur) =>
                    cur?.kind === kind && cur.value === value ? null : { kind, value },
                  )
                }
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
  run, open, busy: engineBusy, onToggle, onRerun, onTag, onDeleted,
}: {
  run: RunRow; open: boolean; busy: boolean;
  onToggle: () => void; onRerun: () => void;
  onTag: (kind: "suite" | "label", value: string) => void;
  onDeleted: () => void;
}) {
  const [detail, setDetail] = useState<RunDetail | null>(null);
  const [confirming, setConfirming] = useState(false);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const [units, setUnits] = useState<LiveUnit[] | null>(null);

  useEffect(() => {
    if (!open) return;
    if (!detail) void api.run(run.id).then(setDetail);
    // Loaded only when a run is expanded: this is every prompt and answer the
    // run produced, which is far too much to fetch for a list of runs.
    if (!units) void api.runUnits(run.id).then((d) => setUnits(d.units)).catch(() => setUnits([]));
  }, [open, detail, units, run.id]);

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
      {/* The tags are their own filters, so the header cannot be one big
          button any more — nesting a button inside a button is invalid and
          React refuses to route the clicks reliably. Expanding now hangs off
          the chevron and the metadata line, both real buttons. */}
      <div className="flex items-center gap-1 pr-2">
        <button
          onClick={onToggle}
          aria-expanded={open}
          title={open ? "Collapse" : "Show this run's items and charts"}
          className="shrink-0 self-stretch px-3 text-faint hover:text-muted"
        >
          <ChevronRight size={13} className={cn("transition-transform", open && "rotate-90")} />
        </button>
        <div className="min-w-0 flex-1 py-2.5">
          <div className="flex flex-wrap items-center gap-1.5">
            {run.suite.split(",").map((name) => name.trim()).filter(Boolean).map((name) => (
              <button
                key={name}
                onClick={() => onTag("suite", name)}
                className="rounded px-1 text-[12px] font-medium hover:bg-accent/12 hover:text-accent"
                title={`Show only runs using ${name}`}
              >
                {name}
              </button>
            ))}
            <Badge tone={tone as never}>{run.status}</Badge>
            {run.label && (
              <button
                onClick={() => onTag("label", run.label)}
                title={`Show only runs labelled ${run.label}`}
                className="rounded-md hover:bg-accent/12"
              >
                <Badge>{run.label}</Badge>
              </button>
            )}
          </div>
          <button onClick={onToggle} className="block w-full text-left">
            <span className="tnum mt-0.5 flex flex-wrap gap-2 text-[11px] text-faint">
              <span>{new Date(run.ts * 1000).toLocaleString()}</span>
              <span>{run.done_units}/{run.planned_units} items</span>
              {run.elapsed_s != null && <span>{run.elapsed_s.toFixed(0)}s</span>}
              <span>{run.models.length} models</span>
            </span>
          </button>
        </div>
        {run.best_score != null && (
          <span className="tnum shrink-0 px-1 text-sm font-semibold">
            {run.best_score.toFixed(2)}
          </span>
        )}
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
              {!!units?.length && (
                <div className="rounded-xl border border-border bg-surface p-4">
                  <h3 className="mb-2 text-sm font-semibold">Every item</h3>
                  <p className="mb-2.5 text-[11px] text-faint">
                    The run replayed in execution order — click a square to read
                    the question and what the model answered.
                  </p>
                  <ItemBrowser units={units} />
                </div>
              )}
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
