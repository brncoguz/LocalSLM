import { useCallback, useEffect, useRef, useState } from "react";
import { Gavel, Loader2 } from "lucide-react";
import { api, streamGet, type JudgeAgreement } from "@/lib/api";
import { Button } from "@/components/ui/primitives";
import { cn } from "@/lib/utils";
import { ChartFrame } from "./ChartFrame";
import { JudgeVerdicts } from "./JudgeVerdicts";

// One pass judges this many items unless the panel is asked for more. Chosen
// against the machine rather than the data: a 9B judge is ~2.5 s per item, and
// a default that ran for ten minutes would make the button feel broken.
const BATCH = 40;

/** How far this run's judge can be trusted.
 *
 *  Led by kappa rather than the agreement rate, and the reason is visible in
 *  the panel itself: these suites are mostly passable, so a judge that
 *  approved everything scores ~89% agreement while knowing nothing. Kappa
 *  subtracts that free credit.
 *
 *  The 2x2 is the part worth looking at. The two off-diagonal cells are not
 *  equally bad — endorsing a wrong answer inflates every score built on the
 *  judge, while rejecting a right one just looks like a hard suite — so they
 *  are coloured differently rather than summed into one error rate. */
export function JudgePanel({ runId }: { runId: string }) {
  const [data, setData] = useState<JudgeAgreement | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [progress, setProgress] = useState<Progress | null>(null);
  const [busy, setBusy] = useState(false);
  const [loaded, setLoaded] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const refresh = useCallback(() => {
    setErr(null);
    return api
      .judgeAgreement(runId)
      .then(setData)
      .catch((e: unknown) => setErr(judgeError(e)))
      .finally(() => setLoaded(true));
  }, [runId]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  useEffect(() => () => abortRef.current?.abort(), []);

  async function judge() {
    setBusy(true);
    setProgress({ done: 0, total: 0, agreed: 0 });
    try {
      const { pending, note } = await api.startJudging(runId, BATCH);
      if (!pending) {
        setProgress(null);
        setErr(note ?? "Nothing left to judge.");
        return;
      }
      const ctrl = new AbortController();
      abortRef.current = ctrl;
      setProgress({ done: 0, total: pending, agreed: 0 });
      setErr(null);
      await streamGet(
        `/runs/${runId}/judge/stream`,
        (ev, d) => {
          if (ev === "verdict") {
            setProgress((p) =>
              p
                ? { ...p, done: p.done + 1, agreed: p.agreed + (d.agreed ? 1 : 0) }
                : p,
            );
          } else if (ev === "failed") {
            setErr(`Judging stopped: ${d.reason}`);
          }
        },
        ctrl.signal,
      );
      await refresh();
    } catch (e: unknown) {
      setErr(judgeError(e));
    } finally {
      setBusy(false);
      setProgress(null);
    }
  }

  const action = (
    <JudgeButton busy={busy} progress={progress} hasData={!!data?.n} onClick={judge} />
  );

  // Swallowing this would make a broken endpoint look like a run nobody had
  // judged yet — the same confusion the rest of this panel exists to prevent,
  // one level up. An older API without the route lands here.
  // Nothing to show until the first fetch settles — but once it has, the panel
  // stays mounted whatever happens. Unmounting it on the way to an error was
  // what made pressing the button look like it did nothing: the panel vanished
  // and came back showing the same message it already had.
  if (!loaded) return null;

  if (!data?.n) {
    return (
      <Shell
        action={action}
        empty={
          err ??
          data?.note ??
          "This run has not been judged yet — score it with the local judge to see how far it agrees with the graders."
        }
      >
        <div />
      </Shell>
    );
  }

  const c = data.confusion!;
  const band = kappaBand(data.kappa ?? null);

  return (
    <ChartFrame
      title="Judge agreement"
      hint={`${data.judge_model} scored this run's answers without seeing the expected answers, and its verdicts are compared against the deterministic graders.`}
      n={data.n}
      minN={1}
    >
      <div className="space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <KappaBar kappa={data.kappa ?? null} band={band} />
          {action}
        </div>
        {err && <p className="text-[11px] text-bad">{err}</p>}

        <div className="grid grid-cols-[auto_1fr_1fr] gap-1 text-[11px]">
          <div />
          <Header>grader: pass</Header>
          <Header>grader: fail</Header>

          <RowLabel>judge: pass</RowLabel>
          <Cell n={c.both_pass} total={data.n} tone="agree" label="agreed" />
          <Cell
            n={c.judge_pass_grader_fail}
            total={data.n}
            tone="bad"
            label="endorsed a wrong answer"
          />

          <RowLabel>judge: fail</RowLabel>
          <Cell
            n={c.judge_fail_grader_pass}
            total={data.n}
            tone="warn"
            label="rejected a right answer"
          />
          <Cell n={c.both_fail} total={data.n} tone="agree" label="agreed" />
        </div>

        <div className="tnum grid grid-cols-2 gap-x-4 gap-y-1 text-[11px] sm:grid-cols-4">
          <Figure label="agreement" value={pct(data.agreement_rate)} muted />
          <Figure label="judge passes" value={pct(data.judge_pass_rate)} />
          <Figure label="grader passes" value={pct(data.grader_pass_rate)} />
          <Figure
            label="judge unreadable"
            value={pct(data.judge_parse_failure_rate)}
            muted={!data.judge_parse_failure_rate}
          />
        </div>

        <RatingSpread histogram={data.rating_histogram} />

        <ByTask byTask={data.by_task} />

        <JudgeVerdicts runId={runId} />
      </div>
    </ChartFrame>
  );
}

/** Where the judge can and cannot be trusted.
 *
 *  The most important thing on this panel, and the reason the headline figure
 *  above it should not be quoted alone. Competence is per task: the same judge
 *  measured here scores 0.88 on exact-match — near perfect — and below zero on
 *  refusal, which is worse than guessing. One number averages an excellent
 *  grader and a useless one into a middling figure describing neither. */
function ByTask({ byTask }: { byTask?: JudgeAgreement["by_task"] }) {
  if (!byTask || Object.keys(byTask).length < 2) return null;
  const rows = Object.entries(byTask).sort(
    ([, a], [, b]) => (b.kappa ?? -Infinity) - (a.kappa ?? -Infinity),
  );

  return (
    <div>
      <p className="mb-1 text-[11px] uppercase tracking-wide text-faint">
        agreement by task
      </p>
      <ul className="space-y-1">
        {rows.map(([task, m]) => {
          const band = kappaBand(m.kappa);
          // Negative kappa has no bar to draw: the judge did worse than
          // guessing, which is a different statement from "scored low".
          const width = m.kappa === null ? 0 : Math.max(0, Math.min(1, m.kappa)) * 100;
          return (
            <li key={task} className="flex items-center gap-2 text-[11px]">
              <span className="w-24 shrink-0 truncate">{task}</span>
              <span className="tnum w-8 shrink-0 text-right text-faint">n={m.n}</span>
              <div className="relative h-1.5 flex-1 overflow-hidden rounded-full bg-surface-2">
                <div
                  className={cn(
                    "absolute top-0 h-full rounded-full",
                    (m.kappa ?? 0) < 0.2 ? "bg-bad" : (m.kappa ?? 0) < 0.6 ? "bg-warn" : "bg-good",
                  )}
                  style={{ width: `${width}%` }}
                />
              </div>
              <span className={cn("tnum w-10 shrink-0 text-right", band.tone)}>
                {m.kappa === null ? "—" : m.kappa.toFixed(2)}
              </span>
              <span className="w-28 shrink-0 truncate text-faint" title={band.label}>
                {m.kappa === null ? "not measurable" : band.label.split(" —")[0]}
              </span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

/** How the judge spent the 1-5 scale.
 *
 *  The most diagnostic view there is. A judge that rates everything 5 is not
 *  grading at all, yet it can still post a respectable agreement rate on a
 *  suite most models pass — the headline number hides exactly the failure this
 *  makes obvious at a glance. */
function RatingSpread({ histogram }: { histogram?: Record<string, number> }) {
  if (!histogram) return null;
  const counts = [1, 2, 3, 4, 5].map((r) => histogram[String(r)] ?? 0);
  const total = counts.reduce((a, b) => a + b, 0);
  if (!total) return null;
  const max = Math.max(...counts);
  const used = counts.filter((n) => n > 0).length;

  return (
    <div>
      <div className="flex items-baseline justify-between gap-2">
        <span className="text-[11px] uppercase tracking-wide text-faint">
          rating spread
        </span>
        {used === 1 && (
          <span className="text-[11px] text-bad">
            every answer got the same rating — this judge is not discriminating
          </span>
        )}
      </div>
      <div className="mt-1 flex items-end gap-1.5">
        {counts.map((n, i) => (
          <div key={i} className="flex-1 text-center" title={`${n} rated ${i + 1}/5`}>
            <div className="flex h-12 items-end">
              <div
                className={cn(
                  "w-full rounded-t",
                  n ? "bg-accent/70" : "bg-surface-2",
                  // 4-5 is a pass under `PASS_AT`, so the split is where the
                  // pass/fail boundary actually falls.
                  i >= 3 && n ? "bg-good/60" : "",
                )}
                style={{ height: `${max ? Math.max(6, (n / max) * 100) : 6}%` }}
              />
            </div>
            <div className="tnum mt-0.5 text-[10px] text-faint">{i + 1}</div>
            <div className="tnum text-[10px] text-muted">{n || ""}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

interface Progress {
  done: number;
  total: number;
  agreed: number;
}

/** The frame every state of this panel shares, so the trigger stays reachable
 *  whether the run has been judged, hasn't been, or failed to load. */
function Shell({
  action,
  empty,
  children,
}: {
  action: React.ReactNode;
  empty: string;
  children: React.ReactNode;
}) {
  return (
    <ChartFrame title="Judge agreement" n={0} minN={1} empty={empty} action={action}>
      {children}
    </ChartFrame>
  );
}

function JudgeButton({
  busy,
  progress,
  hasData,
  onClick,
}: {
  busy: boolean;
  progress: Progress | null;
  hasData: boolean;
  onClick: () => void;
}) {
  if (busy) {
    // Agreement so far is shown live rather than only at the end: it is the
    // number people are waiting for, and watching it settle says more about
    // the judge than a bare item counter.
    const pct = progress?.done && progress.total
      ? Math.round((progress.done / progress.total) * 100)
      : 0;
    return (
      <span className="tnum flex items-center gap-2 text-[11px] text-muted">
        <Loader2 size={12} className="animate-spin" />
        {progress?.total
          ? `judging ${progress.done}/${progress.total} · ${pct}%`
          : "starting…"}
        {!!progress?.done && (
          <span className="text-faint">
            {Math.round((progress.agreed / progress.done) * 100)}% agreed so far
          </span>
        )}
      </span>
    );
  }
  return (
    <Button size="sm" variant="outline" onClick={onClick}>
      <Gavel size={12} />
      {hasData ? `Judge ${BATCH} more` : "Judge this run"}
    </Button>
  );
}

/** Turn the API's structured refusals into something actionable. Each of these
 *  has a different fix, and one generic "could not judge" would send people to
 *  the wrong place. */
function judgeError(e: unknown): string {
  const detail = (e as { detail?: Record<string, string> })?.detail;
  if (detail?.error === "demo_mode") return detail.message;
  if (detail?.error === "judge_missing") return detail.message;
  if (detail?.error === "memory_pressure") return detail.message;
  if (detail?.error === "backend_unavailable") return detail.message;
  if (detail?.error === "engine_busy")
    return "The engine is busy with another run — judging needs it exclusively.";

  // A bare 404 here almost always means the API process predates these routes,
  // because the run itself was found well enough to render everything else on
  // the page. Saying "404" would send someone hunting for a missing run.
  const status = (e as { status?: number })?.status;
  if (status === 404) {
    return (
      "This API server does not have the judge endpoints. Restart it " +
      "(stop and re-run `make serve`) to pick them up."
    );
  }
  return e instanceof Error ? e.message : String(e);
}

/** Landis & Koch's conventional bands. Quoted because a bare kappa means
 *  nothing to most readers, and 0.41 looks like a failing grade when it is
 *  actually moderate agreement. */
function kappaBand(kappa: number | null) {
  if (kappa === null) return { label: "undefined", tone: "text-faint" };
  // Below zero the judge and the grader disagree more often than two people
  // picking at random would. That is a stronger statement than "scored low"
  // and must not be softened into "barely better than guessing".
  if (kappa < 0) return { label: "worse than guessing", tone: "text-bad" };
  if (kappa < 0.2) return { label: "slight — barely better than guessing", tone: "text-bad" };
  if (kappa < 0.4) return { label: "fair", tone: "text-warn" };
  if (kappa < 0.6) return { label: "moderate", tone: "text-warn" };
  if (kappa < 0.8) return { label: "substantial", tone: "text-good" };
  return { label: "almost perfect", tone: "text-good" };
}

function KappaBar({
  kappa,
  band,
}: {
  kappa: number | null;
  band: { label: string; tone: string };
}) {
  // Negative kappa means worse than chance; it exists but has no useful width.
  const pos = kappa === null ? 0 : Math.max(0, Math.min(1, kappa));
  return (
    <div>
      <div className="flex items-baseline justify-between gap-2">
        <span className="text-[11px] uppercase tracking-wide text-faint">
          Cohen&rsquo;s kappa
        </span>
        <span className={`tnum text-lg font-semibold ${band.tone}`}>
          {kappa === null ? "—" : kappa.toFixed(2)}
        </span>
      </div>
      <div className="relative mt-1 h-1.5 overflow-hidden rounded-full bg-surface-2">
        {/* Band boundaries at 0.2 / 0.4 / 0.6 / 0.8, so the number reads
            against the scale rather than in isolation. */}
        {[0.2, 0.4, 0.6, 0.8].map((x) => (
          <div
            key={x}
            className="absolute top-0 h-full w-px bg-border"
            style={{ left: `${x * 100}%` }}
          />
        ))}
        {kappa !== null && (
          <div
            className="absolute top-0 h-full rounded-full bg-accent"
            style={{ width: `${pos * 100}%` }}
          />
        )}
      </div>
      <p className={`mt-1 text-[11px] ${band.tone}`}>
        {band.label}
        {kappa === null && " — one side rated everything the same way"}
      </p>
    </div>
  );
}

function Header({ children }: { children: React.ReactNode }) {
  return (
    <div className="pb-0.5 text-center text-[11px] uppercase tracking-wide text-faint">
      {children}
    </div>
  );
}

function RowLabel({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex items-center pr-2 text-[11px] uppercase tracking-wide text-faint">
      {children}
    </div>
  );
}

const TONES = {
  agree: "border-border bg-surface-2 text-muted",
  bad: "border-bad/30 bg-bad/10 text-bad",
  warn: "border-warn/30 bg-warn/10 text-warn",
} as const;

function Cell({
  n,
  total,
  tone,
  label,
}: {
  n: number;
  total: number;
  tone: keyof typeof TONES;
  label: string;
}) {
  const empty = n === 0;
  return (
    <div
      className={`rounded-md border px-2 py-2 text-center ${
        empty ? "border-border bg-surface-2 text-faint" : TONES[tone]
      }`}
      title={label}
    >
      <div className="tnum text-base font-semibold">{n}</div>
      <div className="text-[11px] opacity-80">
        {total ? `${Math.round((n / total) * 100)}%` : "—"}
      </div>
    </div>
  );
}

function Figure({
  label,
  value,
  muted,
}: {
  label: string;
  value: string;
  muted?: boolean;
}) {
  return (
    <div>
      <div className="text-[11px] uppercase tracking-wide text-faint">{label}</div>
      <div className={muted ? "text-muted" : "font-medium"}>{value}</div>
    </div>
  );
}

function pct(v: number | null | undefined) {
  return v == null ? "—" : `${(v * 100).toFixed(1)}%`;
}
