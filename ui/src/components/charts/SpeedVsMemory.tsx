import {
  CartesianGrid, ResponsiveContainer, Scatter, ScatterChart,
  Tooltip, XAxis, YAxis, ZAxis,
} from "recharts";
import type { ModelSummary } from "@/lib/api";
import { fmt, modelColor } from "@/lib/utils";
import { ChartFrame, axisStyle, gridStyle, tooltipStyle } from "./ChartFrame";

/** Speed against memory — the first real trade-off view.
 *  Up and to the left is better: fast and small. This becomes the full Pareto
 *  frontier once quality scores exist to plot on the Y axis instead. */
export function SpeedVsMemory({ models }: { models: ModelSummary[] }) {
  const data = models
    // Attributed memory, not the machine total: plotting the total put every
    // model at roughly the same X, because most of it was whatever else
    // happened to be resident.
    .filter((m) => m.decode_tps && m.model_vram_bytes)
    .map((m) => ({
      x: m.model_vram_bytes!.max / 1e9,
      y: m.decode_tps!.p50,
      z: m.n,
      model: m.model,
    }));

  return (
    <ChartFrame
      title="Speed vs memory"
      hint="Median throughput against the memory charged to each model. Up and left is better — fast and small."
      n={data.length}
      minN={2}
      empty="Measure at least two models to compare them. Rows recorded before per-runner memory attribution don't carry a per-model figure."
    >
      <ResponsiveContainer width="100%" height={200}>
        <ScatterChart margin={{ top: 8, right: 12, bottom: 14, left: -18 }}>
          <CartesianGrid {...gridStyle} />
          <XAxis
            type="number" dataKey="x" {...axisStyle} unit=" GB"
            label={{ value: "model RAM", position: "insideBottom", offset: -8,
                     fill: "var(--color-faint)", fontSize: 10 }}
          />
          <YAxis type="number" dataKey="y" {...axisStyle} width={38} />
          <ZAxis type="number" dataKey="z" range={[60, 300]} />
          <Tooltip
            {...tooltipStyle}
            cursor={{ strokeDasharray: "3 3", stroke: "var(--color-border)" }}
            content={({ payload }) => {
              const p = payload?.[0]?.payload;
              if (!p) return null;
              return (
                <div className="rounded-lg border border-border bg-surface-2 px-2 py-1.5 text-[11px]">
                  <div style={{ color: modelColor(p.model) }}>{p.model}</div>
                  <div className="tnum text-muted">
                    {p.y.toFixed(1)} tok/s · {fmt.gb(p.x * 1e9)} · n={p.z}
                  </div>
                </div>
              );
            }}
          />
          {data.map((d) => (
            <Scatter key={d.model} data={[d]} fill={modelColor(d.model)} isAnimationActive={false} />
          ))}
        </ScatterChart>
      </ResponsiveContainer>
    </ChartFrame>
  );
}
