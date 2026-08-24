import type { ModelSummary } from "@/lib/api";
import { fmt, modelColor } from "@/lib/utils";
import { ChartFrame } from "./ChartFrame";

/** Time-to-first-answer-token spread per model.
 *
 *  A mean alone is misleading here: a reasoning model can answer instantly on
 *  one prompt and think for a minute on the next, so the range is the finding.
 *  Drawn as a min-p50-p95 range bar on a shared log-ish scale. */
export function LatencyDistribution({ models }: { models: ModelSummary[] }) {
  const data = models.filter((m) => m.ttfat_ms).map((m) => ({ model: m.model, s: m.ttfat_ms! }));
  const max = Math.max(...data.map((d) => d.s.max), 1);

  const pos = (v: number) => `${(Math.log10(Math.max(v, 1)) / Math.log10(max)) * 100}%`;

  return (
    <ChartFrame
      title="Time to first answer token"
      hint="Range across measured requests: min, median, and 95th percentile. Log scale — the spread is often an order of magnitude."
      n={data.length}
      minN={1}
      empty="No latency measurements yet."
    >
      <div className="space-y-3">
        {data.map(({ model, s }) => (
          <div key={model}>
            <div className="tnum flex items-baseline justify-between text-[11px]">
              <span style={{ color: modelColor(model) }}>{model}</span>
              <span className="text-faint">
                median <span className="text-muted">{fmt.ms(s.p50)}</span>
                {s.max > s.min * 3 && <> · up to {fmt.ms(s.max)}</>}
              </span>
            </div>
            <div className="relative mt-1.5 h-1.5 rounded-full bg-surface-2">
              <div
                className="absolute h-full rounded-full opacity-35"
                style={{
                  left: pos(s.min),
                  width: `calc(${pos(s.max)} - ${pos(s.min)})`,
                  background: modelColor(model),
                }}
              />
              <div
                className="absolute top-1/2 h-2.5 w-0.5 -translate-y-1/2 rounded"
                style={{ left: pos(s.p50), background: modelColor(model) }}
                title={`median ${fmt.ms(s.p50)}`}
              />
            </div>
          </div>
        ))}
        <p className="tnum text-[10px] text-faint">log scale · 1 ms → {fmt.ms(max)}</p>
      </div>
    </ChartFrame>
  );
}
