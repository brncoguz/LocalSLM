import type { CellResult } from "@/lib/api";
import { modelColor } from "@/lib/utils";
import { ChartFrame } from "./ChartFrame";

/** Scores with their confidence intervals drawn in.
 *
 *  The interval is the chart, not an annotation on it: at these sample sizes
 *  the bars often overlap completely, and showing only the means would invite
 *  reading a ranking into noise. */
export function ScoreBars({ cells }: { cells: CellResult[] }) {
  const sorted = [...cells].sort((a, b) => b.score - a.score);
  return (
    <ChartFrame
      title="Score by cell"
      hint="Mean score with a 95% bootstrap confidence interval. Overlapping intervals mean the difference isn't established."
      n={cells.length}
      minN={1}
    >
      <div className="space-y-2.5">
        {sorted.map((c) => (
          <div key={c.cell_key}>
            <div className="tnum flex items-baseline justify-between text-[11px]">
              <span style={{ color: modelColor(c.model) }}>{c.label}</span>
              <span className="text-faint">
                <span className="text-muted">{c.score.toFixed(2)}</span>{" "}
                [{c.ci_lo.toFixed(2)}, {c.ci_hi.toFixed(2)}] · n={c.n}
              </span>
            </div>
            <div className="relative mt-1 h-2 rounded-full bg-surface-2">
              {/* interval band */}
              <div
                className="absolute h-full rounded-full opacity-30"
                style={{
                  left: `${Math.max(0, c.ci_lo) * 100}%`,
                  width: `${Math.max(0, Math.min(1, c.ci_hi) - Math.max(0, c.ci_lo)) * 100}%`,
                  background: modelColor(c.model),
                }}
              />
              {/* mean */}
              <div
                className="absolute top-1/2 h-3 w-0.5 -translate-y-1/2 rounded"
                style={{ left: `${c.score * 100}%`, background: modelColor(c.model) }}
              />
            </div>
          </div>
        ))}
        <div className="tnum flex justify-between text-[10px] text-faint">
          <span>0.0</span><span>0.5</span><span>1.0</span>
        </div>
      </div>
    </ChartFrame>
  );
}
