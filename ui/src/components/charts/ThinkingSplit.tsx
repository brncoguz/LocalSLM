import type { ModelSummary } from "@/lib/api";
import { fmt, modelColor } from "@/lib/utils";
import { ChartFrame } from "./ChartFrame";

/** How much of each model's output is internal monologue rather than answer.
 *  Rendered as stacked bars rather than a chart library shape because the
 *  comparison that matters is one ratio per model, and a bar you can read at a
 *  glance beats an axis you have to decode. */
export function ThinkingSplit({ models }: { models: ModelSummary[] }) {
  const data = models
    .filter((m) => m.thinking.thinking_tokens + m.thinking.answer_tokens > 0)
    .map((m) => ({
      model: m.model,
      ratio: m.thinking.ratio ?? 0,
      thinking: m.thinking.thinking_tokens,
      answer: m.thinking.answer_tokens,
    }))
    .sort((a, b) => b.ratio - a.ratio);

  return (
    <ChartFrame
      title="Thinking vs answering"
      hint="Share of generated tokens spent on internal reasoning rather than the reply you see."
      n={data.length}
      minN={1}
      empty="No generations recorded yet."
    >
      <div className="space-y-2.5">
        {data.map((d) => (
          <div key={d.model}>
            <div className="tnum flex items-baseline justify-between text-[11px]">
              <span style={{ color: modelColor(d.model) }}>{d.model}</span>
              <span className="text-faint">
                <span className="text-warn">{fmt.pct(d.ratio)}</span> thinking ·{" "}
                {fmt.int(d.thinking + d.answer)} tokens
              </span>
            </div>
            <div className="mt-1 flex h-1.5 overflow-hidden rounded-full bg-surface-2">
              <div className="bg-warn" style={{ width: `${d.ratio * 100}%` }} />
              <div style={{ width: `${(1 - d.ratio) * 100}%`, background: modelColor(d.model) }} />
            </div>
          </div>
        ))}
      </div>
    </ChartFrame>
  );
}
