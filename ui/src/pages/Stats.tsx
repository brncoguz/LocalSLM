import { useEffect, useState } from "react";
import { RefreshCw } from "lucide-react";
import { api, type Point, type StatsSummary } from "@/lib/api";
import { Button } from "@/components/ui/primitives";
import { TpsTimeline } from "@/components/charts/TpsTimeline";
import { SpeedVsMemory } from "@/components/charts/SpeedVsMemory";
import { ColdWarmBars } from "@/components/charts/ColdWarmBars";
import { ThinkingSplit } from "@/components/charts/ThinkingSplit";
import { LatencyDistribution } from "@/components/charts/LatencyDistribution";
import { fmt, modelColor } from "@/lib/utils";

export function Stats() {
  const [summary, setSummary] = useState<StatsSummary | null>(null);
  const [points, setPoints] = useState<Point[]>([]);
  const [loading, setLoading] = useState(true);

  async function load() {
    setLoading(true);
    try {
      const [s, t] = await Promise.all([api.statsSummary(), api.timeline(1000)]);
      setSummary(s);
      setPoints(t.points);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { void load(); }, []);

  if (!summary) {
    return <div className="p-6 text-sm text-faint">{loading ? "Loading…" : "No data."}</div>;
  }

  const { totals, excluded } = summary;
  const excludedTotal = Object.values(excluded).reduce((a, b) => a + b, 0);

  if (totals.included === 0) {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="max-w-sm text-center">
          <p className="text-sm font-medium">Nothing measured yet</p>
          <p className="mt-1.5 text-[12px] leading-relaxed text-faint">
            Every chat message is recorded as a measurement. Send a few, to more
            than one model, and the comparisons here fill in.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full overflow-y-auto px-5 py-5">
      <div className="mx-auto max-w-5xl space-y-5">
        <header className="flex flex-wrap items-end justify-between gap-3">
          <div>
            <h2 className="text-base font-semibold">Stats</h2>
            <p className="tnum mt-0.5 text-[12px] text-faint">
              {totals.included} of {totals.requests} measurements included ·{" "}
              {summary.models.length} models
            </p>
          </div>
          <Button size="sm" variant="outline" onClick={() => void load()} disabled={loading}>
            <RefreshCw size={12} className={loading ? "animate-spin" : ""} /> Refresh
          </Button>
        </header>

        {/* The exclusion ledger is surfaced, not buried. What got thrown away
            and why is information about the machine, not an implementation detail. */}
        {excludedTotal > 0 && (
          <div className="rounded-lg border border-border bg-surface px-3.5 py-2.5">
            <p className="text-[12px] text-muted">
              <span className="tnum font-medium">{excludedTotal}</span> measurement
              {excludedTotal === 1 ? "" : "s"} excluded from every chart below
            </p>
            <div className="tnum mt-1.5 flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-faint">
              {Object.entries(excluded).map(([reason, n]) => (
                <span key={reason}>
                  <span className="text-muted">{n}</span> {REASONS[reason] ?? reason}
                </span>
              ))}
            </div>
          </div>
        )}

        <div className="grid gap-4 lg:grid-cols-2">
          <div className="lg:col-span-2">
            <TpsTimeline points={points} />
          </div>
          <SpeedVsMemory models={summary.models} />
          <LatencyDistribution models={summary.models} />
          <ColdWarmBars models={summary.models} />
          <ThinkingSplit models={summary.models} />
        </div>

        <ModelTable summary={summary} />
      </div>
    </div>
  );
}

const REASONS: Record<string, string> = {
  warmup: "warmups (discarded by design)",
  swapped: "ran while the machine was swapping",
  memory_pressure: "ran under critical memory pressure",
  unknown: "no reason recorded",
};

/** The numbers behind the charts. A table earns its place here because this is
 *  lookup, not shape — you come to it knowing which model you care about. */
function ModelTable({ summary }: { summary: StatsSummary }) {
  return (
    <div className="rounded-xl border border-border bg-surface">
      <div className="px-4 pt-4">
        <h3 className="text-sm font-semibold">All measurements</h3>
        <p className="mt-0.5 text-[11px] text-faint">
          Medians. p95 in parentheses where it differs meaningfully.
        </p>
      </div>
      <div className="overflow-x-auto p-4 pt-3">
        <table className="tnum w-full min-w-[560px] text-[11px]">
          <thead>
            <tr className="text-left text-faint">
              <th className="pb-2 font-medium">model</th>
              <th className="pb-2 text-right font-medium">n</th>
              <th className="pb-2 text-right font-medium">tok/s</th>
              <th className="pb-2 text-right font-medium">to answer</th>
              <th className="pb-2 text-right font-medium">peak RAM</th>
              <th className="pb-2 text-right font-medium">cold/warm</th>
              <th className="pb-2 text-right font-medium">thinking</th>
            </tr>
          </thead>
          <tbody>
            {summary.models.map((m) => (
              <tr key={m.model} className="border-t border-border">
                <td className="py-1.5" style={{ color: modelColor(m.model) }}>{m.model}</td>
                <td className="py-1.5 text-right text-faint">{m.n}</td>
                <td className="py-1.5 text-right text-muted">
                  {m.decode_tps ? m.decode_tps.p50.toFixed(1) : "—"}
                </td>
                <td className="py-1.5 text-right text-muted">
                  {m.ttfat_ms ? fmt.ms(m.ttfat_ms.p50) : "—"}
                </td>
                <td className="py-1.5 text-right text-muted">
                  {m.peak_rss_bytes ? fmt.gb(m.peak_rss_bytes.max) : "—"}
                </td>
                <td className="py-1.5 text-right text-faint">
                  {m.cold.n}/{m.warm.n}
                </td>
                <td className="py-1.5 text-right text-warn">
                  {m.thinking.ratio != null ? fmt.pct(m.thinking.ratio) : "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
