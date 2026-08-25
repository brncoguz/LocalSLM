import { useEffect, useState } from "react";
import {
  CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { api, type ThinkingPoint } from "@/lib/api";
import { fmt, modelColor } from "@/lib/utils";
import { ChartFrame, axisStyle, gridStyle, tooltipStyle } from "./ChartFrame";

/** Accuracy against reasoning tokens spent — the question the project exists
 *  to answer. A line that goes flat means the extra thinking bought latency
 *  and nothing else. */
export function ThinkingCurve({ runId }: { runId: string }) {
  const [points, setPoints] = useState<ThinkingPoint[] | null>(null);
  const [note, setNote] = useState<string | null>(null);

  useEffect(() => {
    void api.thinking(runId).then((d) => { setPoints(d.points); setNote(d.note); });
  }, [runId]);

  if (!points) return null;
  const byModel = new Map<string, ThinkingPoint[]>();
  for (const p of points) {
    byModel.set(p.model, [...(byModel.get(p.model) ?? []), p]);
  }
  const varied = [...byModel.values()].some((ps) => ps.length > 1);

  if (note || !varied) {
    return (
      <ChartFrame
        title="What reasoning cost"
        n={0}
        minN={1}
        empty={note ?? "Run the same suite at more than one thinking level to plot the trade-off."}
      >
        <div />
      </ChartFrame>
    );
  }

  const rows = points.map((p) => ({
    x: Math.round(p.thinking_tokens),
    [p.model]: p.accuracy,
    think: p.think,
    model: p.model,
    accuracy: p.accuracy,
    ttfat: p.ttfat_ms,
  }));

  return (
    <ChartFrame
      title="What reasoning cost"
      hint="Accuracy against average reasoning tokens spent. A flat line means the extra thinking bought latency and nothing else."
      n={points.reduce((n, p) => n + p.n, 0)}
      minN={1}
    >
      <ResponsiveContainer width="100%" height={190}>
        <LineChart data={rows} margin={{ top: 6, right: 10, bottom: 14, left: -20 }}>
          <CartesianGrid {...gridStyle} />
          <XAxis
            type="number" dataKey="x" {...axisStyle}
            label={{ value: "reasoning tokens", position: "insideBottom", offset: -8,
                     fill: "var(--color-faint)", fontSize: 10 }}
          />
          <YAxis type="number" domain={[0, 1]} {...axisStyle} width={40} />
          <Tooltip
            {...tooltipStyle}
            content={({ payload }) => {
              const p = payload?.[0]?.payload;
              if (!p) return null;
              return (
                <div className="rounded-lg border border-border bg-surface-2 px-2 py-1.5 text-[11px]">
                  <div style={{ color: modelColor(p.model) }}>
                    {p.model} · think {p.think}
                  </div>
                  <div className="tnum text-muted">
                    {(p.accuracy * 100).toFixed(0)}% correct · {p.x} reasoning tokens
                  </div>
                  <div className="tnum text-faint">{fmt.ms(p.ttfat)} to answer</div>
                </div>
              );
            }}
          />
          {[...byModel.keys()].map((m) => (
            <Line key={m} type="monotone" dataKey={m} stroke={modelColor(m)}
              strokeWidth={1.75} dot={{ r: 3, strokeWidth: 0, fill: modelColor(m) }}
              connectNulls isAnimationActive={false} />
          ))}
        </LineChart>
      </ResponsiveContainer>

      <div className="mt-2.5 space-y-1">
        {points.map((p) => (
          <div key={p.cell_key} className="tnum flex flex-wrap items-baseline gap-2 text-[11px]">
            <span style={{ color: modelColor(p.model) }}>{p.model}</span>
            <span className="text-faint">think {p.think}</span>
            <span className="text-muted">{(p.accuracy * 100).toFixed(0)}%</span>
            <span className="text-faint">
              {Math.round(p.thinking_tokens)} reasoning + {Math.round(p.answer_tokens)} answer
              tokens · {fmt.ms(p.ttfat_ms)} to answer
            </span>
            {p.overthink_rate > 0 && (
              <span className="text-warn">
                {(p.overthink_rate * 100).toFixed(0)}% overthought
              </span>
            )}
          </div>
        ))}
        <p className="pt-1 text-[10px] leading-relaxed text-faint">
          "Overthought" counts answers buried under more than three times their own
          length in reasoning.
        </p>
      </div>
    </ChartFrame>
  );
}
