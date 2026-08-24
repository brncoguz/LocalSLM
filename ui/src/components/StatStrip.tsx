import { useState } from "react";
import { ChevronRight, Snowflake } from "lucide-react";
import type { ChatStats } from "@/lib/api";
import { fmt } from "@/lib/utils";

/** The per-message measurement footer.
 *
 *  TTFAT is shown before TTFT on purpose. For a thinking model the first
 *  streamed token is a *thinking* token, so TTFT can be under a second while
 *  the user waits half a minute to see any actual answer. TTFAT is the number
 *  that describes what waiting feels like. */
export function StatStrip({ stats, requestId }: { stats: ChatStats; requestId?: string | null }) {
  const [open, setOpen] = useState(false);
  const think = stats.thinking_chunks;
  const answer = stats.answer_chunks;
  const total = think + answer;
  const thinkPct = total ? think / total : 0;
  const gap = stats.ttfat_ms != null && stats.ttft_ms != null ? stats.ttfat_ms - stats.ttft_ms : 0;

  return (
    <div className="mt-2 text-[11px] text-faint">
      <div className="tnum flex flex-wrap items-center gap-x-3 gap-y-1">
        <span title="Time to first answer token — how long before you see real text, ignoring internal reasoning.">
          <span className="text-muted">{fmt.ms(stats.ttfat_ms)}</span> to answer
        </span>
        <span title="Decode throughput reported by the server.">
          <span className="text-muted">{fmt.tps(stats.decode_tps)}</span> tok/s
        </span>
        <span title="Prompt tokens in / generated tokens out.">
          {fmt.int(stats.prompt_tokens)} in · {fmt.int(stats.output_tokens)} out
        </span>
        {stats.cold && (
          <span className="flex items-center gap-1 text-warn" title="Weights were loaded from disk for this request.">
            <Snowflake size={10} /> cold {fmt.ms(stats.load_ms)}
          </span>
        )}
        <button
          onClick={() => setOpen((v) => !v)}
          className="ml-auto flex items-center gap-0.5 hover:text-muted"
        >
          details <ChevronRight size={11} className={open ? "rotate-90 transition-transform" : "transition-transform"} />
        </button>
      </div>

      {think > 0 && (
        <div className="mt-1.5">
          <div className="flex h-1 overflow-hidden rounded-full bg-surface-2" title={`${think} thinking chunks, ${answer} answer chunks`}>
            <div className="bg-warn" style={{ width: `${thinkPct * 100}%` }} />
            <div className="bg-accent" style={{ width: `${(1 - thinkPct) * 100}%` }} />
          </div>
          <div className="tnum mt-1">
            <span className="text-warn">{fmt.pct(thinkPct)} thinking</span>
            {gap > 250 && <span> · {fmt.ms(gap)} of silence before the first answer token</span>}
          </div>
        </div>
      )}

      {open && (
        <dl className="tnum mt-2 grid grid-cols-2 gap-x-4 gap-y-1 rounded-lg border border-border bg-surface-2 p-2.5 sm:grid-cols-3">
          <Row k="TTFT (any token)" v={fmt.ms(stats.ttft_ms)} />
          <Row k="TTFAT (answer)" v={fmt.ms(stats.ttfat_ms)} />
          <Row k="total" v={fmt.ms(stats.total_ms)} />
          <Row k="prefill" v={`${fmt.tps(stats.prefill_tps)} tok/s`} />
          <Row k="decode" v={`${fmt.tps(stats.decode_tps)} tok/s`} />
          <Row k="jitter (ITL σ)" v={fmt.ms(stats.itl_stddev_ms)} />
          {requestId && <Row k="request" v={<span className="font-mono">{requestId.slice(0, 8)}</span>} />}
        </dl>
      )}
    </div>
  );
}

function Row({ k, v }: { k: string; v: React.ReactNode }) {
  return (
    <div className="flex justify-between gap-2">
      <dt className="text-faint">{k}</dt>
      <dd className="text-muted">{v}</dd>
    </div>
  );
}
