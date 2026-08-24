import { useCallback, useEffect, useRef, useState } from "react";
import { Brain, ChevronDown, CornerDownLeft, Loader2 } from "lucide-react";
import {
  api, streamPost, type ChatStats, type ModelCard,
  type SessionMessage, type SessionRow,
} from "@/lib/api";
import { Button } from "@/components/ui/primitives";
import { StatStrip } from "@/components/StatStrip";
import { SessionSidebar } from "@/components/SessionSidebar";
import { cn, fmt, modelColor } from "@/lib/utils";

type Msg = {
  role: "user" | "assistant";
  content: string;
  thinking?: string;
  stats?: ChatStats;
  requestId?: string | null;
  model?: string;
  error?: string | null;
  streaming?: boolean;
};

const THINK_OPTIONS = [
  { value: "false", label: "off" },
  { value: "low", label: "low" },
  { value: "medium", label: "med" },
  { value: "high", label: "high" },
];

/** Rebuild a stat strip from a stored request row.
 *  Returns null for messages whose request predates a schema change or failed —
 *  the strip is then simply absent rather than showing zeros. */
function statsFromRow(m: SessionMessage): ChatStats | null {
  if (m.total_ms == null) return null;
  return {
    ttft_ms: m.ttft_ms,
    ttfat_ms: m.ttfat_ms,
    total_ms: m.total_ms,
    decode_tps: m.decode_tps,
    prefill_tps: m.prefill_tps,
    prompt_tokens: m.prompt_tokens,
    output_tokens: m.output_tokens,
    thinking_chunks: m.thinking_tokens ?? 0,
    answer_chunks: m.answer_tokens ?? 0,
    think_ratio:
      (m.thinking_tokens ?? 0) + (m.answer_tokens ?? 0) > 0
        ? (m.thinking_tokens ?? 0) / ((m.thinking_tokens ?? 0) + (m.answer_tokens ?? 0))
        : null,
    itl_stddev_ms: m.itl_stddev_ms,
    cold: Boolean(m.cold),
    load_ms: m.srv_load_ms,
  };
}

export function Chat({ models, onMeasured }: { models: ModelCard[]; onMeasured: () => void }) {
  const [model, setModel] = useState<string>("");
  const [think, setThink] = useState("false");
  const [temp, setTemp] = useState(0.7);
  const [input, setInput] = useState("");
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [busy, setBusy] = useState(false);
  const [live, setLive] = useState<{ tps: number | null; elapsed: number } | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [banner, setBanner] = useState<string | null>(null);
  const [sessions, setSessions] = useState<SessionRow[]>([]);
  const endRef = useRef<HTMLDivElement>(null);

  const loadSessions = useCallback(async () => {
    try { setSessions(await api.sessions()); } catch { /* sidebar stays as-is */ }
  }, []);

  useEffect(() => { void loadSessions(); }, [loadSessions]);

  /** Reopen a conversation with the measurements it was recorded with. */
  async function openSession(id: string) {
    try {
      const { messages } = await api.session(id);
      setSessionId(id);
      setBanner(null);
      setMsgs(
        messages.map((m) => ({
          role: m.role,
          content: m.content ?? "",
          thinking: m.thinking ?? undefined,
          stats: m.role === "assistant" ? (statsFromRow(m) ?? undefined) : undefined,
          requestId: m.request_id,
          model: m.model ?? undefined,
          error: m.error,
        })),
      );
      const row = sessions.find((s) => s.id === id);
      if (row?.model && models.some((mm) => mm.name === row.model)) setModel(row.model);
    } catch {
      setBanner("Couldn't load that conversation.");
    }
  }

  function newChat() {
    setSessionId(null);
    setMsgs([]);
    setBanner(null);
  }

  useEffect(() => {
    if (!model && models.length) setModel(models[0].name);
  }, [models, model]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [msgs, live]);

  const current = models.find((m) => m.name === model);
  const canThink = true; // capability is per-model; the API records "unsupported" when it isn't

  async function send() {
    const text = input.trim();
    if (!text || busy || !model) return;
    setInput("");
    setBanner(null);
    setBusy(true);
    setLive({ tps: null, elapsed: 0 });
    setMsgs((m) => [
      ...m,
      { role: "user", content: text },
      { role: "assistant", content: "", thinking: "", model, streaming: true },
    ]);

    const patch = (fn: (m: Msg) => Msg) =>
      setMsgs((all) => all.map((m, i) => (i === all.length - 1 ? fn(m) : m)));

    try {
      await streamPost(
        "/chat",
        {
          model,
          message: text,
          session_id: sessionId,
          think: think === "false" ? false : think,
          temperature: temp,
        },
        (ev, data) => {
          if (ev === "meta") setSessionId(data.session_id);
          else if (ev === "thinking") patch((m) => ({ ...m, thinking: (m.thinking ?? "") + data.delta }));
          else if (ev === "token") patch((m) => ({ ...m, content: m.content + data.delta }));
          else if (ev === "stat") setLive({ tps: data.tps, elapsed: data.elapsed_ms });
          else if (ev === "done") {
            patch((m) => ({
              ...m,
              content: data.text || m.content,
              thinking: data.thinking || m.thinking,
              stats: data.stats,
              requestId: data.request_id,
              error: data.error,
              streaming: false,
            }));
            onMeasured();
            void loadSessions();
          }
        },
      );
    } catch (e: any) {
      const d = e?.detail;
      if (d?.error === "model_missing") {
        setBanner(d.message ?? "That model isn't installed any more.");
      } else if (d?.error === "benchmark_running") {
        setBanner(
          `A benchmark is running (${d.run_id ?? "unknown"}). Chat is blocked so its timings stay clean.`,
        );
      } else {
        setBanner(e?.message ? `Request failed: ${e.message}` : "Request failed.");
      }
      setMsgs((all) => all.slice(0, -1));
    } finally {
      setBusy(false);
      setLive(null);
    }
  }

  return (
    <div className="flex h-full">
      <SessionSidebar
        sessions={sessions}
        activeId={sessionId}
        onSelect={(id) => void openSession(id)}
        onNew={newChat}
        onRename={async (id, title) => {
          await api.renameSession(id, title);
          void loadSessions();
        }}
        onDelete={async (id) => {
          await api.deleteSession(id);
          if (id === sessionId) newChat();
          void loadSessions();
        }}
      />
      <div className="flex min-w-0 flex-1 flex-col">
      {/* Header: model identity + decoding controls */}
      <div className="flex flex-wrap items-center gap-3 border-b border-border px-4 py-2.5">
        <div className="flex items-center gap-2">
          <span
            className="h-2.5 w-2.5 rounded-full"
            style={{ background: model ? modelColor(model) : "var(--color-faint)" }}
          />
          <div className="relative">
            <select
              value={model}
              onChange={(e) => setModel(e.target.value)}
              className="appearance-none rounded-lg border border-border bg-surface py-1.5 pl-2.5 pr-7 text-sm font-medium focus:outline-none focus:ring-2 focus:ring-accent/50"
            >
              {models.map((m) => (
                <option key={m.name} value={m.name}>
                  {m.name}
                </option>
              ))}
            </select>
            <ChevronDown size={13} className="pointer-events-none absolute right-2 top-2.5 text-faint" />
          </div>
          {current && (
            <span className="tnum text-[11px] text-faint">
              {current.quant} · {fmt.gb(current.size_bytes)} ·{" "}
              <span className={current.state === "warm" ? "text-good" : ""}>{current.state}</span>
            </span>
          )}
        </div>

        <div className="ml-auto flex items-center gap-3">
          <label className="flex items-center gap-1.5 text-[11px] text-faint">
            <Brain size={12} />
            thinking
            <div className="flex overflow-hidden rounded-md border border-border">
              {THINK_OPTIONS.map((o) => (
                <button
                  key={o.value}
                  disabled={!canThink}
                  onClick={() => setThink(o.value)}
                  className={cn(
                    "px-1.5 py-0.5 text-[11px] transition-colors",
                    think === o.value ? "bg-warn/15 text-warn" : "text-faint hover:text-muted",
                  )}
                >
                  {o.label}
                </button>
              ))}
            </div>
          </label>
          <label className="tnum flex items-center gap-1.5 text-[11px] text-faint">
            temp
            <input
              type="range" min={0} max={1} step={0.1} value={temp}
              onChange={(e) => setTemp(Number(e.target.value))}
              className="w-16 accent-[var(--color-accent)]"
            />
            <span className="w-6 text-muted">{temp.toFixed(1)}</span>
          </label>
        </div>
      </div>

      {banner && (
        <div className="border-b border-warn/25 bg-warn/8 px-4 py-2 text-[12px] text-warn">{banner}</div>
      )}

      {/* Transcript */}
      <div className="flex-1 overflow-y-auto px-4 py-5">
        <div className="mx-auto max-w-3xl space-y-5">
          {msgs.length === 0 && <EmptyState model={model} />}
          {msgs.map((m, i) =>
            m.role === "user" ? (
              <div key={i} className="flex justify-end">
                <div className="max-w-[85%] rounded-2xl rounded-br-sm bg-surface-2 px-3.5 py-2 text-sm">
                  {m.content}
                </div>
              </div>
            ) : (
              <div key={i} className="max-w-[92%]">
                {m.thinking ? <Thinking text={m.thinking} streaming={m.streaming} /> : null}
                <div className="whitespace-pre-wrap text-sm leading-relaxed">
                  {m.content}
                  {m.streaming && !m.content && (
                    <span className="inline-flex items-center gap-1.5 text-faint">
                      <Loader2 size={12} className="animate-spin" />
                      {m.thinking ? "thinking…" : "waiting for first token…"}
                    </span>
                  )}
                </div>
                {m.error && <div className="mt-1 text-xs text-bad">{m.error}</div>}
                {m.stats && <StatStrip stats={m.stats} requestId={m.requestId} />}
              </div>
            ),
          )}
          <div ref={endRef} />
        </div>
      </div>

      {/* Composer */}
      <div className="border-t border-border px-4 py-3">
        <div className="mx-auto max-w-3xl">
          {live && (
            <div className="tnum mb-1.5 flex items-center gap-3 text-[11px] text-faint">
              <Loader2 size={11} className="animate-spin" />
              {(live.elapsed / 1000).toFixed(1)}s
              {live.tps != null && <span>· {live.tps.toFixed(1)} chunks/s</span>}
            </div>
          )}
          <div className="flex items-end gap-2 rounded-xl border border-border bg-surface p-2">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); void send(); }
              }}
              rows={1}
              placeholder={model ? `Message ${model}…` : "No models installed"}
              disabled={!model}
              className="max-h-40 flex-1 resize-none bg-transparent px-1.5 py-1 text-sm placeholder:text-faint focus:outline-none"
            />
            <Button size="sm" onClick={() => void send()} disabled={busy || !input.trim() || !model}>
              {busy ? <Loader2 size={13} className="animate-spin" /> : <CornerDownLeft size={13} />}
            </Button>
          </div>
          <p className="mt-1.5 text-[11px] text-faint">
            Every message is recorded as a measurement — the numbers here are the ones the report uses.
          </p>
        </div>
      </div>
      </div>
    </div>
  );
}

function Thinking({ text, streaming }: { text: string; streaming?: boolean }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="mb-2">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-1.5 rounded-md bg-warn/10 px-2 py-0.5 text-[11px] text-warn hover:bg-warn/15"
      >
        <Brain size={11} className={streaming ? "pulse" : ""} />
        {streaming ? "thinking…" : `thought · ${text.length.toLocaleString()} chars`}
        <ChevronDown size={11} className={open ? "rotate-180 transition-transform" : "transition-transform"} />
      </button>
      {open && (
        <pre className="mt-1.5 max-h-72 overflow-y-auto whitespace-pre-wrap rounded-lg border border-border bg-surface-2 p-2.5 font-mono text-[11px] leading-relaxed text-muted">
          {text}
        </pre>
      )}
    </div>
  );
}

function EmptyState({ model }: { model: string }) {
  return (
    <div className="pt-16 text-center">
      <p className="text-sm text-muted">Chat with {model || "a local model"}.</p>
      <p className="mx-auto mt-2 max-w-md text-[12px] leading-relaxed text-faint">
        Try the same prompt with thinking <span className="text-warn">off</span> and{" "}
        <span className="text-warn">high</span> — on a small model, reasoning often costs an order
        of magnitude in latency for no change in the answer.
      </p>
    </div>
  );
}
