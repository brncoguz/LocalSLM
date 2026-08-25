import { useEffect, useState } from "react";
import { api, type ThinkingPoint } from "@/lib/api";
import { fmt, modelColor } from "@/lib/utils";
import { ChartFrame } from "./ChartFrame";

/** The gap between the first token and the first *readable* token.
 *
 *  Public benchmarks still quote TTFT for reasoning models as though it meant
 *  something. On these models it can be under a second while the user waits
 *  half a minute, and the paired bars make that gap the whole picture. */
export function TtftVsTtfat({ runId }: { runId: string }) {
  const [points, setPoints] = useState<ThinkingPoint[] | null>(null);
  useEffect(() => { void api.thinking(runId).then((d) => setPoints(d.points)); }, [runId]);
  if (!points?.length) return null;

  const usable = points.filter((p) => p.ttft_ms != null && p.ttfat_ms != null);
  if (!usable.length) return null;
  const max = Math.max(...usable.map((p) => p.ttfat_ms!));

  return (
    <ChartFrame
      title="First token vs first answer"
      hint="TTFT is when anything arrives; TTFAT is when readable text arrives. For a reasoning model the difference is what the user actually waits through."
      n={usable.reduce((n, p) => n + p.n, 0)}
      minN={1}
    >
      <div className="space-y-3">
        {usable.map((p) => {
          const gap = p.ttfat_ms! - p.ttft_ms!;
          return (
            <div key={p.cell_key}>
              <div className="tnum flex flex-wrap items-baseline justify-between gap-2 text-[11px]">
                <span>
                  <span style={{ color: modelColor(p.model) }}>{p.model}</span>
                  <span className="ml-1.5 text-faint">think {p.think}</span>
                </span>
                <span className="text-faint">
                  {fmt.ms(p.ttft_ms)} → <span className="text-muted">{fmt.ms(p.ttfat_ms)}</span>
                  {gap > 250 && <span className="ml-1 text-warn">+{fmt.ms(gap)} silent</span>}
                </span>
              </div>
              <div className="mt-1 flex h-2 overflow-hidden rounded-full bg-surface-2">
                <div style={{ width: `${(p.ttft_ms! / max) * 100}%`, background: modelColor(p.model) }} />
                <div className="bg-warn/60" style={{ width: `${(gap / max) * 100}%` }} />
              </div>
            </div>
          );
        })}
        <p className="text-[10px] text-faint">
          Solid is time to the first token; amber is the silence before any readable answer.
        </p>
      </div>
    </ChartFrame>
  );
}
