import { Line, LineChart, ResponsiveContainer, YAxis } from "recharts";

/** Inline trend for a model card. Deliberately axis-free: it shows shape, not
 *  values, and anything precise belongs in a real chart. */
export function Sparkline({
  values, color, height = 22,
}: {
  values: number[];
  color: string;
  height?: number;
}) {
  if (values.length < 2) return null;
  const data = values.map((v, i) => ({ i, v }));
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 2, right: 0, bottom: 2, left: 0 }}>
        <YAxis hide domain={["dataMin", "dataMax"]} />
        <Line
          type="monotone" dataKey="v" stroke={color} strokeWidth={1.25}
          dot={false} isAnimationActive={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
