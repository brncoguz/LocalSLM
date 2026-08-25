import { useEffect, useState } from "react";
import { AlertTriangle, Split } from "lucide-react";
import { api, type DisagreementItem, type Disagreements } from "@/lib/api";
import { cn, modelColor } from "@/lib/utils";

/** Where the signal is.
 *
 *  On a suite where everything scores around 0.9, a handful of items carry all
 *  the discriminating information and the rest are noise. And an item every
 *  model fails usually means the item is wrong — a bad gold label or an
 *  ambiguous prompt — so this doubles as a tool for fixing the benchmark. */
export function FailureGallery({ runId }: { runId?: string }) {
  const [data, setData] = useState<Disagreements | null>(null);
  const [tab, setTab] = useState<"split" | "failed">("split");

  useEffect(() => { void api.disagreements(runId ?? "").then(setData); }, [runId]);
  if (!data) return null;

  const items = tab === "split" ? data.disagreements : data.all_failed;

  return (
    <div className="rounded-xl border border-border bg-surface p-4">
      <div className="mb-3 flex flex-wrap items-center gap-3">
        <div>
          <h3 className="text-sm font-semibold">Where the models differ</h3>
          <p className="tnum mt-0.5 text-[11px] text-faint">
            {data.n_items} items · {data.n_disagreements} split ·{" "}
            {data.n_all_failed} failed by everyone
          </p>
        </div>
        <div className="ml-auto flex overflow-hidden rounded-md border border-border">
          <Tab active={tab === "split"} onClick={() => setTab("split")}
            icon={<Split size={11} />} label={`Disagreements ${data.n_disagreements}`} />
          <Tab active={tab === "failed"} onClick={() => setTab("failed")}
            icon={<AlertTriangle size={11} />} label={`All failed ${data.n_all_failed}`} />
        </div>
      </div>

      {tab === "failed" && data.all_failed.length > 0 && (
        <p className="mb-2.5 rounded-md bg-warn/8 px-2.5 py-1.5 text-[11px] text-warn">
          When every model fails the same item, suspect the item first — a wrong
          expected answer or an ambiguous prompt.
        </p>
      )}

      {items.length === 0 ? (
        <p className="rounded-lg border border-dashed border-border px-3 py-6 text-center text-[11px] text-faint">
          {tab === "split"
            ? "No disagreements yet — run a benchmark across two or more models."
            : "Nothing failed by every model."}
        </p>
      ) : (
        <div className="space-y-2">
          {items.map((it) => <Row key={`${it.run_id}-${it.item_id}`} item={it} />)}
        </div>
      )}
    </div>
  );
}

function Tab({ active, onClick, icon, label }: {
  active: boolean; onClick: () => void; icon: React.ReactNode; label: string;
}) {
  return (
    <button onClick={onClick}
      className={cn("flex items-center gap-1.5 px-2 py-1 text-[11px] transition-colors",
        active ? "bg-surface-2 text-fg" : "text-faint hover:text-muted")}>
      {icon}{label}
    </button>
  );
}

function Row({ item }: { item: DisagreementItem }) {
  const gold = item.models[0]?.gold;
  return (
    <div className="rounded-lg border border-border bg-surface-2 px-2.5 py-2">
      <div className="tnum flex flex-wrap items-center gap-2 text-[11px]">
        <span className="font-mono text-muted">{item.item_id}</span>
        <span className="text-faint">{item.task}</span>
        {gold && <span className="text-faint">expected <span className="text-good">{gold}</span></span>}
      </div>
      <div className="mt-1.5 flex flex-wrap gap-1.5">
        {item.models.map((m) => (
          <span key={m.model + m.request_id}
            title={m.pred ? `answered: ${m.pred}` : undefined}
            className={cn("tnum flex items-center gap-1 rounded px-1.5 py-0.5 text-[11px]",
              m.passed ? "bg-good/10" : "bg-bad/10")}>
            <span style={{ color: modelColor(m.model) }}>{m.model}</span>
            <span className={m.passed ? "text-good" : "text-bad"}>
              {m.pred ?? (m.passed ? "ok" : "fail")}
            </span>
          </span>
        ))}
      </div>
    </div>
  );
}
