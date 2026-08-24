import { useCallback, useRef, useState } from "react";
import {
  AlertTriangle, Check, Download, Flame, Loader2, Search,
  Snowflake, Trash2, X, Zap,
} from "lucide-react";
import {
  api, pullModel, type CatalogEntry, type ModelCard,
  type ModelsResponse, type PullProgress,
} from "@/lib/api";
import { Badge, Button, Card, Stat } from "@/components/ui/primitives";
import { ModelDetail } from "@/components/ModelDetail";
import { fmt, modelColor } from "@/lib/utils";

type PullState = { progress: PullProgress | null; error: string | null; abort: AbortController };

/** Cards, not a table. Each card is a scorecard: what the model is, and what it
 *  actually did on this machine — so "which should I use" is answerable here. */
export function Models({ data, refresh }: { data: ModelsResponse | null; refresh: () => void }) {
  const [busy, setBusy] = useState<string | null>(null);
  const [pulls, setPulls] = useState<Record<string, PullState>>({});
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);
  const [custom, setCustom] = useState("");
  const [detail, setDetail] = useState<string | null>(null);
  const pullsRef = useRef(pulls);
  pullsRef.current = pulls;

  const startPull = useCallback(
    async (name: string) => {
      if (!name.trim() || pullsRef.current[name]) return;
      const abort = new AbortController();
      setPulls((p) => ({ ...p, [name]: { progress: null, error: null, abort } }));
      try {
        await pullModel(
          name,
          (progress) => setPulls((p) => (p[name] ? { ...p, [name]: { ...p[name], progress } } : p)),
          abort.signal,
        );
        setPulls((p) => { const { [name]: _, ...rest } = p; return rest; });
        refresh();
      } catch (e: any) {
        if (abort.signal.aborted) {
          setPulls((p) => { const { [name]: _, ...rest } = p; return rest; });
          return;
        }
        const d = e?.detail;
        const msg =
          d?.error === "disk_guard"
            ? `Needs ${fmt.gb(d.needed_bytes)} but only ${fmt.gb(d.free_bytes)} free — the guard keeps ${fmt.gb(d.guard_bytes)} in reserve.`
            : d?.error === "benchmark_running"
              ? "A benchmark is running. Pulling would compete for disk and skew its timings."
              : (e?.message ?? "Pull failed.");
        setPulls((p) => ({ ...p, [name]: { ...p[name], error: msg } }));
      }
    },
    [refresh],
  );

  const dismiss = (name: string) =>
    setPulls((p) => { const { [name]: _, ...rest } = p; return rest; });

  async function act(name: string, fn: () => Promise<unknown>) {
    setBusy(name);
    try { await fn(); refresh(); } finally { setBusy(null); setConfirmDelete(null); }
  }

  if (!data) return <div className="p-6 text-sm text-faint">Loading…</div>;

  const measured = data.installed.filter((m) => m.headline?.decode_tps);
  const fastest = measured.length
    ? measured.reduce((a, b) => (a.headline!.decode_tps! > b.headline!.decode_tps! ? a : b))
    : null;
  const used = data.installed.reduce((n, m) => n + m.size_bytes, 0);

  return (
    <div className="h-full overflow-y-auto px-5 py-5">
      <div className="mx-auto max-w-5xl space-y-6">
        <header className="flex flex-wrap items-end justify-between gap-3">
          <div>
            <h2 className="text-base font-semibold">Models</h2>
            <p className="tnum mt-0.5 text-[12px] text-faint">
              {data.installed.length} installed · {fmt.gb(used)} of weights ·{" "}
              {fmt.gb(data.disk.free_bytes)} disk free
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
            <ModelTile
              key={m.name}
              m={m}
              busy={busy === m.name}
              confirming={confirmDelete === m.name}
              onOpen={() => setDetail(m.name)}
              onConfirm={() => setConfirmDelete(m.name)}
              onCancel={() => setConfirmDelete(null)}
              act={act}
            />
          ))}
        </div>

        <section>
          <h3 className="text-sm font-semibold">Worth pulling</h3>
          <p className="mt-0.5 max-w-2xl text-[12px] leading-relaxed text-faint">
            Your installed models are mostly Qwen-architecture Q4 thinking models. Comparing
            them measures size, not architecture. These add a real contrast axis.
          </p>

          <div className="mt-3 space-y-2">
            {data.catalog.filter((c) => !c.installed).map((c) => (
              <CatalogRow
                key={c.name}
                c={c}
                pull={pulls[c.name]}
                onPull={() => startPull(c.name)}
                onDismiss={() => dismiss(c.name)}
              />
            ))}
          </div>

          {/* Anything on the Ollama registry, not just the curated list. */}
          <div className="mt-4 flex items-center gap-2">
            <div className="flex flex-1 items-center gap-2 rounded-lg border border-border bg-surface px-2.5 py-1.5">
              <Search size={13} className="shrink-0 text-faint" />
              <input
                value={custom}
                onChange={(e) => setCustom(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && custom.trim()) { void startPull(custom.trim()); setCustom(""); }
                }}
                placeholder="Any model from the Ollama registry, e.g. mistral:7b"
                className="w-full bg-transparent font-mono text-[12px] placeholder:font-sans placeholder:text-faint focus:outline-none"
              />
            </div>
            <Button
              size="sm"
              disabled={!custom.trim() || !!pulls[custom.trim()]}
              onClick={() => { void startPull(custom.trim()); setCustom(""); }}
            >
              <Download size={13} /> Pull
            </Button>
          </div>

          {/* Pulls started from the free-text box have no catalog row to live in. */}
          {Object.entries(pulls)
            .filter(([n]) => !data.catalog.some((c) => c.name === n))
            .map(([name, st]) => (
              <div key={name} className="mt-2 rounded-lg border border-border bg-surface px-3 py-2.5">
                <div className="flex items-center gap-2">
                  <span className="font-mono text-[12px]">{name}</span>
                  <div className="ml-auto">
                    <PullControls pull={st} onPull={() => startPull(name)} onDismiss={() => dismiss(name)} />
                  </div>
                </div>
                <PullBar pull={st} />
              </div>
            ))}
        </section>
      </div>
      {detail && <ModelDetail name={detail} onClose={() => setDetail(null)} />}
    </div>
  );
}

function ModelTile({
  m, busy, confirming, onOpen, onConfirm, onCancel, act,
}: {
  m: ModelCard; busy: boolean; confirming: boolean;
  onOpen: () => void; onConfirm: () => void; onCancel: () => void;
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
              <button onClick={onOpen} className="truncate text-sm font-semibold hover:underline">
                {m.name}
              </button>
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
            <Button size="icon" variant="ghost" title="Delete these weights"
              disabled={busy} onClick={onConfirm}>
              {busy ? <Loader2 size={13} className="animate-spin" /> : <Trash2 size={13} />}
            </Button>
          </div>
        </div>

        {confirming && (
          <div className="mt-3 flex items-center gap-2 rounded-lg border border-bad/30 bg-bad/8 px-2.5 py-2 text-[11px]">
            <AlertTriangle size={13} className="shrink-0 text-bad" />
            <span className="text-muted">
              Delete {fmt.gb(m.size_bytes)} of weights? Past measurements are kept.
            </span>
            <div className="ml-auto flex gap-1">
              <Button size="sm" variant="ghost" onClick={onCancel}>Cancel</Button>
              <Button size="sm" variant="danger" onClick={() => act(m.name, () => api.remove(m.name))}>
                Delete
              </Button>
            </div>
          </div>
        )}

        <div className="mt-3.5 grid grid-cols-3 gap-3 border-t border-border pt-3">
          {h?.decode_tps ? (
            <>
              <Stat label="tok/s" value={h.decode_tps.toFixed(1)} color={color} sub={`n=${h.n}`}
                hint="Mean decode throughput across measured requests." />
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

function CatalogRow({
  c, pull, onPull, onDismiss,
}: {
  c: CatalogEntry; pull?: PullState; onPull: () => void; onDismiss: () => void;
}) {
  return (
    <div className="rounded-lg border border-border bg-surface px-3 py-2.5">
      <div className="flex items-start gap-3">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <span className="font-mono text-[12px] font-medium">{c.name}</span>
            <span className="tnum text-[11px] text-faint">{c.size_gb} GB</span>
            {!c.fits && <Badge tone="bad">not enough disk</Badge>}
          </div>
          <p className="mt-0.5 text-[11px] leading-relaxed text-faint">{c.why}</p>
        </div>
        <div className="shrink-0">
          <PullControls pull={pull} disabled={!c.fits} onPull={onPull} onDismiss={onDismiss} />
        </div>
      </div>
      <PullBar pull={pull} />
    </div>
  );
}

function PullControls({
  pull, disabled, onPull, onDismiss,
}: {
  pull?: PullState; disabled?: boolean; onPull: () => void; onDismiss: () => void;
}) {
  if (pull?.error) {
    return (
      <div className="flex gap-1">
        <Button size="sm" variant="outline" onClick={onPull}>Retry</Button>
        <Button size="icon" variant="ghost" onClick={onDismiss}><X size={13} /></Button>
      </div>
    );
  }
  if (pull) {
    return (
      <Button size="sm" variant="ghost" onClick={() => pull.abort.abort()} title="Cancel this pull">
        <Loader2 size={12} className="animate-spin" /> Cancel
      </Button>
    );
  }
  return (
    <Button size="sm" variant="outline" disabled={disabled} onClick={onPull}>
      <Download size={13} /> Pull
    </Button>
  );
}

function PullBar({ pull }: { pull?: PullState }) {
  if (!pull) return null;
  if (pull.error) {
    return (
      <div className="mt-2 flex items-start gap-1.5 rounded-md bg-bad/8 px-2 py-1.5 text-[11px] text-bad">
        <AlertTriangle size={12} className="mt-px shrink-0" />
        <span>{pull.error}</span>
      </div>
    );
  }
  const p = pull.progress;
  const pct = p?.pct ?? 0;
  const verifying = p && p.pct != null && p.pct >= 99.9;
  return (
    <div className="mt-2">
      <div className="h-1 overflow-hidden rounded-full bg-surface-2">
        <div
          className="h-full bg-accent transition-[width] duration-300"
          style={{ width: `${pct}%` }}
        />
      </div>
      <div className="tnum mt-1 flex items-center gap-2 text-[11px] text-faint">
        {verifying ? (
          <span className="flex items-center gap-1 text-good"><Check size={11} /> verifying…</span>
        ) : (
          <>
            <span className="text-muted">{pct.toFixed(0)}%</span>
            {p && p.total > 0 && <span>{fmt.gb(p.completed)} / {fmt.gb(p.total)}</span>}
            {p?.bytes_per_s ? <span>{(p.bytes_per_s / 1e6).toFixed(1)} MB/s</span> : null}
            {p?.eta_s ? <span className="ml-auto">~{p.eta_s}s left</span> : null}
          </>
        )}
      </div>
    </div>
  );
}
