import { useState } from "react";
import { CheckCircle2, ChevronLeft, ChevronRight, XCircle, Zap } from "lucide-react";
import { cn, fmt, modelColor } from "@/lib/utils";

/** One graded item, however it arrived.
 *
 *  The live stream emits these as a run executes and `/runs/{id}/units`
 *  replays a finished one in the same shape, so the browser below cannot tell
 *  the two apart — which is the point. A run being watched and the same run
 *  read back a week later should not be able to disagree about what happened. */
export interface LiveUnit {
  done: number;
  total: number;
  cell: string;
  model: string;
  suite: string;
  item: string;
  score: number | null;
  passed: boolean;
  decode_tps: number | null;
  ttfat_ms: number | null;
  prompt: string;
  answer: string;
  expected: string;
  thinking_tokens: number;
  excluded: boolean;
  elapsed_s: number;
}

/** Every item in a run, as a strip of squares you can click through.
 *
 *  The strip is a timeline: streaks are the signal, and five reds in a row from
 *  one model is a different story from five scattered across the run. No
 *  average shows that difference, which is why the accuracy figure beside it is
 *  not a substitute.
 *
 *  `live` keeps the card following the newest item until you pin one. Without
 *  that, clicking item 12 mid-run would be overwritten a second later by item
 *  13 — the inspection would be impossible to perform on a moving run. */
export function ItemBrowser({
  units,
  live = false,
}: {
  units: LiveUnit[];
  live?: boolean;
}) {
  // Keyed on the unit's index in the run rather than its position in the
  // strip: the strip keeps only a recent window, so an array index would
  // quietly start pointing at a different item as the run advanced.
  const [pinned, setPinned] = useState<number | null>(null);
  if (!units.length) return null;

  const latest = units[units.length - 1];
  const fallback = live ? latest : units[0];
  const shown = pinned === null ? fallback : (units.find((u) => u.done === pinned) ?? fallback);
  const following = live && pinned === null;

  const failures = units.filter((u) => !u.passed && !u.excluded);
  const jump = (delta: number) => {
    const i = units.findIndex((u) => u.done === shown.done);
    const next = units[Math.min(units.length - 1, Math.max(0, i + delta))];
    if (next) setPinned(next.done);
  };
  const nextFailure = () => {
    const after = failures.find((u) => u.done > shown.done) ?? failures[0];
    if (after) setPinned(after.done);
  };

  return (
    <div className="space-y-2">
      <Heartbeat units={units} selected={shown.done} onSelect={setPinned} />

      <div className="tnum flex flex-wrap items-center gap-2 text-[11px]">
        {following ? (
          <span className="flex items-center gap-1.5 text-faint">
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-accent" />
            following live — click any square to inspect it
          </span>
        ) : (
          <span className="text-faint">
            item {shown.done} of {shown.total}
            {live && (
              <button
                onClick={() => setPinned(null)}
                className="ml-2 text-accent hover:underline"
              >
                follow live
              </button>
            )}
          </span>
        )}
        <span className="ml-auto flex items-center gap-1">
          <StepButton onClick={() => jump(-1)} disabled={shown.done <= units[0].done}>
            <ChevronLeft size={12} />
          </StepButton>
          <StepButton onClick={() => jump(1)} disabled={shown.done >= latest.done}>
            <ChevronRight size={12} />
          </StepButton>
          {failures.length > 0 && (
            <button
              onClick={nextFailure}
              className="rounded px-1.5 py-0.5 text-bad hover:bg-bad/10"
              title="Jump to the next failing item"
            >
              next fail ({failures.length})
            </button>
          )}
        </span>
      </div>

      <CurrentItem unit={shown} />
    </div>
  );
}

/** The last stretch of results as one dot per item.
 *
 *  Streaks are the signal here — five reds in a row from one model is a
 *  different story from five scattered across the run, and no average shows
 *  the difference. */
function Heartbeat({
  units,
  selected,
  onSelect,
}: {
  units: LiveUnit[];
  selected: number;
  onSelect: (done: number) => void;
}) {
  const recent = units.slice(-80);
  if (!recent.length) return null;
  return (
    <div className="flex flex-wrap gap-[3px]">
      {recent.map((u) => (
        <button
          key={u.done}
          onClick={() => onSelect(u.done)}
          title={`${u.suite}/${u.item} · ${u.model} · ${
            u.excluded ? "excluded" : u.passed ? "pass" : "fail"
          } — click to inspect`}
          aria-label={`Item ${u.item}, ${u.passed ? "passed" : "failed"}`}
          className={cn(
            "h-2.5 w-2.5 rounded-[2px] transition-transform hover:scale-125",
            // Excluded is neither pass nor fail: the measurement was thrown
            // out, and colouring it as a failure would blame the model for the
            // machine swapping.
            u.excluded ? "bg-border" : u.passed ? "bg-good/80" : "bg-bad/80",
            u.done === selected && "ring-2 ring-fg/70 ring-offset-1 ring-offset-surface",
          )}
        />
      ))}
    </div>
  );
}

function StepButton({
  onClick,
  disabled,
  children,
}: {
  onClick: () => void;
  disabled: boolean;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className="rounded p-0.5 text-faint hover:bg-surface-2 hover:text-muted disabled:opacity-30 disabled:hover:bg-transparent"
    >
      {children}
    </button>
  );
}

/** The question in flight, with what came back. */
function CurrentItem({ unit }: { unit: LiveUnit }) {
  return (
    <div className="rounded-lg border border-border bg-surface-2 p-2.5">
      <div className="tnum mb-1.5 flex flex-wrap items-center gap-2 text-[11px]">
        <span style={{ color: modelColor(unit.model) }} className="font-medium">
          {unit.model}
        </span>
        <span className="text-faint">
          {unit.suite}/{unit.item}
        </span>
        {unit.thinking_tokens > 0 && (
          <span className="flex items-center gap-1 text-warn">
            <Zap size={10} />
            {unit.thinking_tokens} thinking
          </span>
        )}
        {unit.decode_tps && (
          <span className="text-faint">{unit.decode_tps.toFixed(0)} tok/s</span>
        )}
        {unit.ttfat_ms != null && (
          <span className="text-faint">{fmt.ms(unit.ttfat_ms)} to answer</span>
        )}
        <span className="ml-auto flex items-center gap-1">
          {unit.excluded ? (
            <span className="text-faint">excluded — machine swapped</span>
          ) : unit.passed ? (
            <>
              <CheckCircle2 size={12} className="text-good" />
              <span className="text-good">pass</span>
            </>
          ) : (
            <>
              <XCircle size={12} className="text-bad" />
              <span className="text-bad">fail</span>
            </>
          )}
        </span>
      </div>

      <p className="text-[11px] leading-relaxed text-muted">{unit.prompt}</p>

      <div className="mt-1.5 border-l-2 border-border pl-2">
        <p className="whitespace-pre-wrap text-[11px] leading-relaxed text-fg">
          {unit.answer.trim() || (
            <span className="text-faint">
              {unit.thinking_tokens > 0
                ? "(no answer — the budget went on reasoning)"
                : "(no answer)"}
            </span>
          )}
        </p>
      </div>

      {unit.expected && !unit.passed && (
        <p className="tnum mt-1.5 text-[11px] text-faint">
          expected <span className="text-muted">{unit.expected}</span>
        </p>
      )}
    </div>
  );
}
