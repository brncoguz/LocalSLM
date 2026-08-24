import { useState } from "react";
import { Flame, Snowflake, Zap } from "lucide-react";
import { api, type CatalogEntry, type ModelCard, type ModelsResponse } from "@/lib/api";
import { Badge, Button, Card, Stat } from "@/components/ui/primitives";
import { fmt, modelColor } from "@/lib/utils";

/** Cards, not a table. Each card is a scorecard: what the model is, and what
 *  it actually did on this machine — so "which should I use" is answerable
 *  from this screen alone. */
export function Models({ data, refresh }: { data: ModelsResponse | null; refresh: () => void }) {
  const [busy, setBusy] = useState<string | null>(null);
  if (!data) return <div className="p-6 text-sm text-faint">Loading…</div>;

  async function act(name: string, fn: () => Promise<unknown>) {
    setBusy(name);
    try { await fn(); refresh(); } finally { setBusy(null); }
  }

  const measured = data.installed.filter((m) => m.headline?.decode_tps);
  const fastest = measured.length
    ? measured.reduce((a, b) => (a.headline!.decode_tps! > b.headline!.decode_tps! ? a : b))
    : null;

  return (
    <div className="h-full overflow-y-auto px-5 py-5">
      <div className="mx-auto max-w-5xl space-y-6">
        <header className="flex flex-wrap items-end justify-between gap-3">
          <div>
            <h2 className="text-base font-semibold">Models</h2>
            <p className="mt-0.5 text-[12px] text-faint">
              {data.installed.length} installed · {fmt.gb(data.disk.free_bytes)} disk free
            </p>
          </div>
          {fastest && (
            <div className="tnum text-[11px] text-faint">
              fastest measured:{" "}
              <span style={{ color: modelColor(fastest.name) }}>{fastest.name}</span>{" "}
              {fastest.headline!.decode_tps!.toFixed(1)} tok/s
            </div>
          )}
        </header>

        <div className="grid gap-3 sm:grid-cols-2">
          {data.installed.map((m) => (
            <ModelTile key={m.name} m={m} busy={busy === m.name} act={act} />
          ))}
        </div>

        <section>
          <h3 className="text-sm font-semibold">Worth pulling</h3>
          <p className="mt-0.5 max-w-2xl text-[12px] leading-relaxed text-faint">
            Your installed models are all Qwen-architecture Q4 thinking models — two of them from
            the same family. Comparing them mostly measures size, not architecture. These add a
            real contrast axis.
          </p>
          <div className="mt-3 space-y-2">
            {data.catalog.filter((c) => !c.installed).map((c) => (
              <CatalogRow key={c.name} c={c} />
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}

function ModelTile({
  m, busy, act,
}: {
  m: ModelCard;
  busy: boolean;
  act: (n: string, fn: () => Promise<unknown>) => Promise<void>;
}) {
  const color = modelColor(m.name);
  const h = m.headline;
  return (
    <Card className="overflow-hidden">
      <div className="h-0.5" style={{ background: color }} />
      <div className="p-4">
        <div className="flex items-start justify-between gap-2">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <h3 className="truncate text-sm font-semibold">{m.name}</h3>
              {m.state === "warm" ? (
                <Badge tone="good"><Flame size={9} /> warm</Badge>
              ) : (
                <Badge><Snowflake size={9} /> cold</Badge>
              )}
            </div>
            <div className="tnum mt-1 flex flex-wrap gap-1.5 text-[11px] text-faint">
              <Badge>{m.quant}</Badge>
              {m.params_b && <Badge>{m.params_b.toFixed(1)}B</Badge>}
              <Badge>{fmt.gb(m.size_bytes)}</Badge>
              {m.family && <Badge>{m.family}</Badge>}
            </div>
          </div>
          <div className="flex shrink-0 gap-1">
            {m.state === "warm" ? (
              <Button size="icon" variant="ghost" title="Unload from memory"
                disabled={busy} onClick={() => act(m.name, () => api.unload(m.name))}>
                <Snowflake size={13} />
              </Button>
            ) : (
              <Button size="icon" variant="ghost" title="Load into memory"
                disabled={busy} onClick={() => act(m.name, () => api.warm(m.name))}>
                <Zap size={13} />
              </Button>
            )}
          </div>
        </div>

        <div className="mt-3.5 grid grid-cols-3 gap-3 border-t border-border pt-3">
          {h?.decode_tps ? (
            <>
              <Stat label="tok/s" value={h.decode_tps.toFixed(1)} color={color}
                sub={`n=${h.n}`} hint="Mean decode throughput across measured requests." />
              <Stat label="to answer" value={fmt.ms(h.ttfat_ms)}
                hint="Mean time to first answer token — excludes internal reasoning." />
              <Stat label="peak RAM" value={fmt.gb(h.peak_rss_bytes)}
                hint="Peak resident memory of the Ollama runner while generating." />
            </>
          ) : (
            <div className="col-span-3 text-[11px] text-faint">
              Not measured yet — send it a message and its numbers appear here.
            </div>
          )}
        </div>
      </div>
    </Card>
  );
}

function CatalogRow({ c }: { c: CatalogEntry }) {
  return (
    <div className="flex items-start gap-3 rounded-lg border border-border bg-surface px-3 py-2.5">
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <span className="font-mono text-[12px] font-medium">{c.name}</span>
          <span className="tnum text-[11px] text-faint">{c.size_gb} GB</span>
          {!c.fits && <Badge tone="bad">not enough disk</Badge>}
        </div>
        <p className="mt-0.5 text-[11px] leading-relaxed text-faint">{c.why}</p>
      </div>
      <code className="shrink-0 rounded bg-surface-2 px-2 py-1 text-[11px] text-muted">
        ollama pull {c.name}
      </code>
    </div>
  );
}

