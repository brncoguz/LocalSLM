import {
  Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer,
  Tooltip, XAxis, YAxis,
} from "recharts";
import type { ModelSummary } from "@/lib/api";
import { fmt, modelColor } from "@/lib/utils";
import { ChartFrame, axisStyle, gridStyle, tooltipStyle } from "./ChartFrame";

/** The cold-start tax: what you pay on the first message after a model has
 *  been evicted. Only models measured in both states can answer this, so the
 *  rest are excluded rather than shown as zero. */
export function ColdWarmBars({ models }: { models: ModelSummary[] }) {
  const data = models
    .filter((m) => m.cold.load_ms && m.warm.load_ms)
    .map((m) => ({
      model: m.model,
      cold: m.cold.load_ms!.p50,
      warm: m.warm.load_ms!.p50,
      factor: m.cold.load_ms!.p50 / Math.max(m.warm.load_ms!.p50, 1),
    }));

  return (
    <ChartFrame
      title="Cold-start penalty"
      hint="Median load time when weights had to come from disk, versus when the model was already resident."
      n={data.length}
      minN={1}
      empty="Needs a model measured both cold and warm. Unload one, then chat with it again."
    >
      <ResponsiveContainer width="100%" height={Math.max(120, data.length * 56)}>
        <BarChart data={data} layout="vertical" margin={{ top: 0, right: 44, bottom: 0, left: 4 }}>
          <CartesianGrid {...gridStyle} vertical horizontal={false} />
          <XAxis type="number" {...axisStyle} unit=" ms" />
          <YAxis
            type="category" dataKey="model" {...axisStyle} width={104}
            tick={{ fontSize: 10, fill: "var(--color-muted)" }}
          />
          <Tooltip
            {...tooltipStyle}
            cursor={{ fill: "var(--color-surface-2)" }}
            formatter={(v, k) => [fmt.ms(Number(v)), String(k)]}
          />
          <Bar dataKey="warm" name="warm" radius={[0, 3, 3, 0]} barSize={9}>
            {data.map((d) => (
              <Cell key={d.model} fill="var(--color-border)" />
            ))}
          </Bar>
          <Bar dataKey="cold" name="cold" radius={[0, 3, 3, 0]} barSize={9}>
            {data.map((d) => (
              <Cell key={d.model} fill={modelColor(d.model)} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <div className="tnum mt-2 space-y-0.5 text-[11px] text-faint">
        {data.map((d) => (
          <div key={d.model}>
            <span style={{ color: modelColor(d.model) }}>{d.model}</span> pays{" "}
            <span className="text-muted">{d.factor.toFixed(1)}×</span> on a cold start
          </div>
        ))}
      </div>
    </ChartFrame>
  );
}
