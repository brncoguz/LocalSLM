import { useEffect, useState } from "react";
import { api, type DeterminismCell } from "@/lib/api";
import { ChartFrame } from "./ChartFrame";
import { modelColor } from "@/lib/utils";

/** How often repeated sampling of the same prompt gives the same answer.
 *
 *  Presented as a measurement rather than a check. It is tempting to assume
 *  temperature 0 is deterministic and treat a deviation as a bug, but that is
 *  a property of the hardware and runtime worth measuring, not asserting. */
export function DeterminismPanel({ runId }: { runId: string }) {
  const [cells, setCells] = useState<DeterminismCell[] | null>(null);
  const [note, setNote] = useState<string | null>(null);

  useEffect(() => {
    void api.determinism(runId).then((d) => { setCells(d.cells); setNote(d.note); });
  }, [runId]);

  if (!cells) return null;
  if (note) {
    return (
      <ChartFrame title="Output stability" n={0} minN={1} empty={note}>
        <div />
      </ChartFrame>
    );
  }

  return (
    <ChartFrame
      title="Output stability"
      hint="Share of repeated samples that produced the same answer. 1.00 means every sample was identical."
      n={cells.reduce((n, c) => n + c.n_items, 0)}
      minN={1}
    >
      <div className="space-y-3">
        {cells.map((c) => {
          const wobbly = Object.entries(c.items)
            .filter(([, m]) => !m.fully_deterministic)
            .sort((a, b) => a[1].exact_duplicate_rate - b[1].exact_duplicate_rate);
          // Score variance is the part that matters: identical wording is nice,
          // but flipping between right and wrong on the same prompt is the risk.
          const flipped = wobbly.filter(([, m]) => (m.score_stdev ?? 0) > 0);
          return (
            <div key={c.cell_key}>
              <div className="tnum flex flex-wrap items-baseline justify-between gap-2 text-[11px]">
                <span>
                  <span style={{ color: modelColor(c.model) }}>{c.model}</span>
                  <span className="ml-1.5 text-faint">
                    temp {c.temperature}{c.think !== "off" && ` · think ${c.think}`}
                  </span>
                </span>
                <span className="text-faint">
                  <span className="text-muted">
                    {(c.mean_duplicate_rate * 100).toFixed(0)}%
                  </span>{" "}
                  identical · {c.fully_deterministic_items}/{c.n_items} items stable ·{" "}
                  {c.samples_per_item} samples
                </span>
              </div>
              <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-surface-2">
                <div className="h-full rounded-full"
                  style={{
                    width: `${c.mean_duplicate_rate * 100}%`,
                    background: modelColor(c.model),
                  }} />
              </div>
              {flipped.length > 0 && (
                <p className="tnum mt-1 text-[10px] text-warn">
                  {flipped.length} item{flipped.length === 1 ? "" : "s"} changed from correct
                  to incorrect between samples
                </p>
              )}
            </div>
          );
        })}
      </div>
    </ChartFrame>
  );
}
