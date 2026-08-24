import {
  CartesianGrid, Legend, Line, LineChart, ResponsiveContainer,
  Tooltip, XAxis, YAxis,
} from "recharts";
import type { Point } from "@/lib/api";
import { modelColor } from "@/lib/utils";
import { ChartFrame, axisStyle, gridStyle, tooltipStyle } from "./ChartFrame";

/** Decode throughput over time, one line per model in its identity colour.
 *  Points are indexed by measurement order rather than wall-clock: sessions are
 *  bursty, and a real time axis would compress every burst into a spike. */
export function TpsTimeline({ points }: { points: Point[] }) {
  const models = [...new Set(points.map((p) => p.model))];
  const withTps = points.filter((p) => p.decode_tps != null);

  const rows: Record<string, number | string>[] = [];
  const seen: Record<string, number> = {};
  for (const p of withTps) {
    seen[p.model] = (seen[p.model] ?? 0) + 1;
    rows.push({ i: rows.length, [p.model]: p.decode_tps!, label: p.model });
  }

  return (
    <ChartFrame
      title="Throughput over time"
      hint="Decode tokens per second for each measured request, in the order they happened."
      n={withTps.length}
      minN={2}
      empty="Chat with a model to start plotting throughput."
    >
      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={rows} margin={{ top: 4, right: 8, bottom: 0, left: -18 }}>
          <CartesianGrid {...gridStyle} />
          <XAxis dataKey="i" {...axisStyle} tick={false} label={undefined} />
          <YAxis {...axisStyle} width={38} unit="" />
          <Tooltip
            {...tooltipStyle}
            formatter={(v, name) => [`${Number(v).toFixed(1)} tok/s`, String(name)]}
            labelFormatter={() => ""}
          />
          <Legend
            iconType="plainline"
            wrapperStyle={{ fontSize: 11, paddingTop: 4 }}
          />
          {models.map((m) => (
            <Line
              key={m}
              type="monotone"
              dataKey={m}
              stroke={modelColor(m)}
              strokeWidth={1.75}
              dot={{ r: 2.5, strokeWidth: 0, fill: modelColor(m) }}
              connectNulls
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </ChartFrame>
  );
}
