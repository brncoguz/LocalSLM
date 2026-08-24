import type { ReactNode } from "react";
import { Info } from "lucide-react";

/** Shared chrome for every chart: title, the sample size, a one-line
 *  explanation, and a real empty state.
 *
 *  `n` is not decoration. With a handful of measurements most apparent
 *  differences are noise, so the count sits next to the title where it can't
 *  be missed, and `minN` blocks the chart entirely until there's enough to
 *  look at. */
export function ChartFrame({
  title, hint, n, minN = 1, empty, children, action,
}: {
  title: string;
  hint?: string;
  n?: number;
  minN?: number;
  empty?: string;
  children: ReactNode;
  action?: ReactNode;
}) {
  const enough = n === undefined || n >= minN;
  return (
    <div className="rounded-xl border border-border bg-surface p-4">
      <div className="mb-3 flex items-start gap-2">
        <div className="min-w-0">
          <h3 className="flex items-center gap-1.5 text-sm font-semibold">
            {title}
            {hint && (
              <span title={hint} className="cursor-help text-faint">
                <Info size={11} />
              </span>
            )}
          </h3>
          {n !== undefined && (
            <p className="tnum mt-0.5 text-[11px] text-faint">
              {n} measurement{n === 1 ? "" : "s"}
              {n > 0 && n < 5 && " · too few to read as a trend"}
            </p>
          )}
        </div>
        {action && <div className="ml-auto shrink-0">{action}</div>}
      </div>
      {enough ? (
        children
      ) : (
        <div className="flex h-40 items-center justify-center rounded-lg border border-dashed border-border text-center">
          <p className="max-w-[24ch] text-[12px] leading-relaxed text-faint">
            {empty ?? "Not enough measurements yet."}
          </p>
        </div>
      )}
    </div>
  );
}

/** Recharts tooltip styled to match the app rather than its own defaults. */
export const tooltipStyle = {
  contentStyle: {
    background: "var(--color-surface-2)",
    border: "1px solid var(--color-border)",
    borderRadius: "8px",
    fontSize: "11px",
    padding: "6px 8px",
  },
  labelStyle: { color: "var(--color-muted)", marginBottom: 2 },
  itemStyle: { padding: 0 },
} as const;

export const axisStyle = {
  stroke: "var(--color-faint)",
  fontSize: 10,
  tickLine: false,
} as const;

export const gridStyle = {
  stroke: "var(--color-border)",
  strokeDasharray: "2 4",
  vertical: false,
} as const;
