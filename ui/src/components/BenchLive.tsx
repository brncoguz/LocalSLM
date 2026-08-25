import { fmt, modelColor } from "@/lib/utils";
import { ItemBrowser, type LiveUnit } from "./ItemBrowser";

export type { LiveUnit };

/** What a benchmark looks like while it runs.
 *
 *  A counter tells you a run is progressing; it does not tell you whether it is
 *  going well, and the questions are the interesting part. Watching the actual
 *  prompts and answers stream past is the fastest way to notice a broken item,
 *  a model answering in the wrong format, or a suite that is far too easy —
 *  things the final score reports as a number long after you could have stopped
 *  the run.
 *
 *  The tallies are deliberately live rather than saved for the end. A cell that
 *  is clearly losing is worth seeing at item 12, not item 200. */
export function BenchLive({
  units,
  onStop,
}: {
  units: LiveUnit[];
  onStop?: () => void;
}) {
  const latest = units[units.length - 1];
  const pct = latest.total ? (latest.done / latest.total) * 100 : 0;
  const rate = latest.elapsed_s > 0 ? latest.done / latest.elapsed_s : 0;
  const remaining = rate > 0 ? (latest.total - latest.done) / rate : null;

  return (
    <div className="mt-3.5 space-y-3 border-t border-border pt-3">
      <div>
        <div className="h-1.5 overflow-hidden rounded-full bg-surface-2">
          <div
            className="h-full bg-accent transition-[width] duration-300"
            style={{ width: `${pct}%` }}
          />
        </div>
        <div className="tnum mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px]">
          <span className="font-medium text-muted">
            {latest.done}/{latest.total}
          </span>
          <span className="text-faint">{pct.toFixed(0)}%</span>
          <span className="text-faint">{fmt.duration(latest.elapsed_s)} elapsed</span>
          {remaining !== null && remaining > 1 && (
            <span className="text-faint">~{fmt.duration(remaining)} left</span>
          )}
          {rate > 0 && <span className="text-faint">{rate.toFixed(2)} items/s</span>}
          {onStop && (
            <button onClick={onStop} className="ml-auto text-faint hover:text-bad">
              stop
            </button>
          )}
        </div>
      </div>

      <ItemBrowser units={units} live />

      <Scoreboard units={units} />
    </div>
  );
}

/** Per-cell accuracy and throughput, accumulated from the stream so the board
 *  moves with every item rather than appearing once the run has finished. */
function Scoreboard({ units }: { units: LiveUnit[] }) {
  const cells = new Map<
    string,
    { n: number; passed: number; tps: number[]; model: string }
  >();
  for (const u of units) {
    // An excluded measurement is neither a pass nor a failure — counting it
    // either way would blame the model for the machine swapping.
    if (u.excluded) continue;
    const c = cells.get(u.cell) ?? { n: 0, passed: 0, tps: [], model: u.model };
    c.n += 1;
    if (u.passed) c.passed += 1;
    if (u.decode_tps) c.tps.push(u.decode_tps);
    cells.set(u.cell, c);
  }
  const board = [...cells.entries()].sort(
    (a, b) => b[1].passed / b[1].n - a[1].passed / a[1].n,
  );
  if (!board.length) return null;

  return (
    <div className="space-y-1">
      <p className="text-[11px] uppercase tracking-wide text-faint">
        running score · updates every item
      </p>
      {board.map(([cell, c]) => {
        const acc = c.passed / c.n;
        const tps = c.tps.length ? c.tps.reduce((a, b) => a + b, 0) / c.tps.length : null;
        return (
          <div key={cell} className="flex items-center gap-2 text-[11px]">
            <span
              className="w-44 shrink-0 truncate"
              style={{ color: modelColor(c.model) }}
              title={cell}
            >
              {cell}
            </span>
            <div className="relative h-1.5 flex-1 overflow-hidden rounded-full bg-surface-2">
              <div
                className="absolute top-0 h-full rounded-full transition-[width] duration-300"
                style={{ width: `${acc * 100}%`, background: modelColor(c.model) }}
              />
            </div>
            <span className="tnum w-10 shrink-0 text-right text-muted">
              {(acc * 100).toFixed(0)}%
            </span>
            <span className="tnum w-12 shrink-0 text-right text-faint">
              {tps ? `${tps.toFixed(0)} t/s` : "—"}
            </span>
            <span className="tnum w-8 shrink-0 text-right text-faint">n={c.n}</span>
          </div>
        );
      })}
    </div>
  );
}
