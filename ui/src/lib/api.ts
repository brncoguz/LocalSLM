/** Typed client. Shapes mirror the Pydantic/sqlite rows on the backend;
 *  when they drift, this file is the one place to fix. */

export interface Health {
  ok: boolean;
  runtime: { ok: boolean; version?: string; error?: string; endpoint: string };
  host: {
    chip: string | null; cores: number; ram_gb: number; macos: string;
    python: string; ollama_version: string | null; env_hash: string;
  };
  config_hash: string;
  loaded: { name: string; size_bytes: number; size_vram_bytes: number; expires_at: string | null }[];
  memory: {
    free_bytes: number | null; runner_rss_bytes: number | null;
    swap_used_bytes: number | null; pressure: "normal" | "warn" | "critical" | "unknown";
  };
  disk: { free_bytes: number };
  power: { on_ac: boolean; battery_pct: number | null };
  db: { requests: number; runs: number };
  lease: { busy: boolean; holder: string | null; exclusive: boolean; run_id: string | null };
  presets: string[];
}

export interface Headline {
  n: number; decode_tps: number | null; ttfat_ms: number | null; peak_rss_bytes: number | null;
}

export interface ModelCard {
  name: string; digest: string | null; family: string | null; quant: string | null;
  params_b: number | null; size_bytes: number; state: "warm" | "cold";
  vram_bytes: number | null; expires_at: string | null; headline: Headline | null;
}

export interface CatalogEntry {
  name: string; size_gb: number; family: string; why: string;
  installed: boolean; fits: boolean;
}

export interface ModelsResponse {
  installed: ModelCard[];
  disk: { free_bytes: number };
  catalog: CatalogEntry[];
}

export interface ModelDetail {
  name: string; arch: string | null; family: string | null; quant: string | null;
  params_b: number | null; ctx_max: number | null;
  capabilities: {
    completion: boolean; thinking: boolean; tools: boolean;
    vision: boolean; schema: boolean; think_levels: string[];
  };
  details: Record<string, unknown>;
  recent: Record<string, unknown>[];
}

export interface PullProgress {
  status: string;
  completed: number;
  total: number;
  pct: number | null;
  bytes_per_s: number | null;
  eta_s: number | null;
}

export interface ChatStats {
  ttft_ms: number | null;
  ttfat_ms: number | null;
  total_ms: number;
  decode_tps: number | null;
  prefill_tps: number | null;
  prompt_tokens: number | null;
  output_tokens: number | null;
  thinking_chunks: number;
  answer_chunks: number;
  think_ratio: number | null;
  itl_stddev_ms: number | null;
  cold: boolean;
  load_ms: number | null;
}

async function json<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(`/api${path}`, init);
  if (!r.ok) {
    let detail: unknown;
    try { detail = (await r.json()).detail; } catch { detail = await r.text(); }
    throw Object.assign(new Error(`${r.status}`), { status: r.status, detail });
  }
  return r.json() as Promise<T>;
}

export const api = {
  health: () => json<Health>("/health"),
  models: () => json<ModelsResponse>("/models"),
  modelDetail: (n: string) => json<ModelDetail>(`/models/${n}/detail`),
  warm: (n: string) => json<{ load_ms: number }>(`/models/${n}/warm`, { method: "POST" }),
  unload: (n: string) => json<{ ok: boolean }>(`/models/${n}/unload`, { method: "POST" }),
  remove: (n: string) => json<{ freed_bytes: number }>(`/models/${n}`, { method: "DELETE" }),
  sessions: () => json<{ id: string; title: string; model: string; messages: number }[]>("/sessions"),
  request: (id: string) => json<Record<string, unknown>>(`/requests/${id}`),
};

/** Pull a model, reporting aggregate progress. Resolves when the pull finishes
 *  or rejects with the server's reason (disk guard, benchmark running, 404). */
export function pullModel(
  name: string,
  onProgress: (p: PullProgress) => void,
  signal?: AbortSignal,
): Promise<void> {
  return new Promise((resolve, reject) => {
    streamPost(
      "/models/pull",
      { name },
      (ev, data) => {
        if (ev === "progress") onProgress(data as PullProgress);
        else if (ev === "done") resolve();
        // The stream carries its own errors: a mid-pull disk-guard abort is a
        // 200 response that fails, so it can't be surfaced by status code.
        else if (ev === "error") reject(new Error(data.message ?? "pull failed"));
      },
      signal,
    ).then(
      () => resolve(),
      (e) => reject(e),
    );
  });
}

export type SSEHandler = (event: string, data: any) => void;

/** POST that returns an SSE stream. EventSource can't POST, so we parse the
 *  framing ourselves — which we'd need anyway to stamp arrival times. */
export async function streamPost(
  path: string,
  body: unknown,
  onEvent: SSEHandler,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(`/api${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok) {
    let detail: any;
    try { detail = (await res.json()).detail; } catch { detail = null; }
    throw Object.assign(new Error(`${res.status}`), { status: res.status, detail });
  }
  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const frames = buf.split("\n\n");
    buf = frames.pop() ?? "";
    for (const frame of frames) {
      let name = "message";
      const dataLines: string[] = [];
      for (const line of frame.split("\n")) {
        if (line.startsWith("event: ")) name = line.slice(7).trim();
        else if (line.startsWith("data: ")) dataLines.push(line.slice(6));
      }
      if (dataLines.length) {
        try { onEvent(name, JSON.parse(dataLines.join("\n"))); } catch { /* keep streaming */ }
      }
    }
  }
}
