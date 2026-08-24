import { useEffect, useState } from "react";
import { X } from "lucide-react";
import { api, type ModelDetail as Detail, type ModelSummary, type Point } from "@/lib/api";
import { Badge, Stat } from "@/components/ui/primitives";
import { Sparkline } from "@/components/charts/Sparkline";
import { fmt, modelColor } from "@/lib/utils";

/** Drill-down for one model: what it is, and its own measurement history.
 *  A side sheet rather than a route, so you never lose your place in the grid. */
export function ModelDetail({ name, onClose }: { name: string; onClose: () => void }) {
  const [detail, setDetail] = useState<Detail | null>(null);
  const [history, setHistory] = useState<{ points: Point[]; summary: ModelSummary } | null>(null);
  const color = modelColor(name);

  useEffect(() => {
    let alive = true;
    void (async () => {
      const [d, h] = await Promise.allSettled([api.modelDetail(name), api.modelHistory(name)]);
      if (!alive) return;
      if (d.status === "fulfilled") setDetail(d.value);
      if (h.status === "fulfilled") setHistory(h.value);
    })();
    return () => { alive = false; };
  }, [name]);

  const s = history?.summary;
  const tps = history?.points.map((p) => p.decode_tps).filter((v): v is number => v != null) ?? [];

  return (
    <div className="fixed inset-0 z-50 flex justify-end" onClick={onClose}>
      <div className="absolute inset-0 bg-black/50" />
      <div
        className="relative flex h-full w-full max-w-md flex-col overflow-y-auto border-l border-border bg-surface"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="h-0.5 shrink-0" style={{ background: color }} />
        <div className="flex items-start justify-between gap-2 px-4 pt-4">
          <div className="min-w-0">
            <h2 className="truncate text-sm font-semibold" style={{ color }}>{name}</h2>
            {detail && (
              <p className="tnum mt-0.5 text-[11px] text-faint">
                {detail.arch} · {detail.params_b?.toFixed(1)}B · {detail.quant} ·{" "}
                {fmt.int(detail.ctx_max)} ctx
              </p>
            )}
          </div>
          <button onClick={onClose} className="rounded p-1 text-faint hover:bg-surface-2 hover:text-fg">
            <X size={15} />
          </button>
        </div>

        {detail && (
          <div className="flex flex-wrap gap-1.5 px-4 pt-3">
            {Object.entries(detail.capabilities)
              .filter(([k, v]) => v === true && k !== "completion")
              .map(([k]) => <Badge key={k} tone="accent">{k}</Badge>)}
            {detail.capabilities.think_levels.length > 0 && (
              <Badge>think: {detail.capabilities.think_levels.join(" / ")}</Badge>
            )}
          </div>
        )}

        <div className="space-y-4 p-4">
          {s && s.n > 0 ? (
            <>
              <div className="grid grid-cols-3 gap-3 rounded-lg border border-border bg-surface-2 p-3">
                <Stat label="tok/s" value={s.decode_tps ? s.decode_tps.p50.toFixed(1) : "—"}
                  sub={s.decode_tps ? `p95 ${s.decode_tps.p95.toFixed(1)}` : undefined} color={color} />
                <Stat label="to answer" value={s.ttfat_ms ? fmt.ms(s.ttfat_ms.p50) : "—"}
                  sub={s.ttfat_ms ? `max ${fmt.ms(s.ttfat_ms.max)}` : undefined} />
                <Stat label="peak RAM" value={s.peak_rss_bytes ? fmt.gb(s.peak_rss_bytes.max) : "—"} />
              </div>

              {tps.length > 1 && (
                <div>
                  <p className="mb-1 text-[11px] text-faint">throughput trend · {tps.length} requests</p>
                  <Sparkline values={tps} color={color} height={40} />
                </div>
              )}

              <dl className="tnum space-y-1.5 text-[11px]">
                <Row k="measurements" v={`${s.n}`} />
                <Row k="cold / warm" v={`${s.cold.n} / ${s.warm.n}`} />
                {s.cold.load_ms && s.warm.load_ms && (
                  <Row
                    k="cold-start penalty"
                    v={`${(s.cold.load_ms.p50 / Math.max(s.warm.load_ms.p50, 1)).toFixed(1)}× · ${fmt.ms(s.cold.load_ms.p50)}`}
                  />
                )}
                {s.thinking.ratio != null && (
                  <Row k="thinking share" v={fmt.pct(s.thinking.ratio)} />
                )}
                {s.prefill_tps && <Row k="prefill" v={`${s.prefill_tps.p50.toFixed(0)} tok/s`} />}
              </dl>
            </>
          ) : (
            <p className="rounded-lg border border-dashed border-border px-3 py-6 text-center text-[11px] text-faint">
              No measurements yet. Chat with this model and its history appears here.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

function Row({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex justify-between border-b border-border pb-1.5">
      <dt className="text-faint">{k}</dt>
      <dd className="text-muted">{v}</dd>
    </div>
  );
}
