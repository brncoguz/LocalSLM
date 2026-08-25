/** Typed client. Shapes mirror the Pydantic/sqlite rows on the backend;
 *  when they drift, this file is the one place to fix. */

import { demoGet, demoRefuse, isDemo } from "@/lib/demo";

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

export interface Spread {
  n: number; mean: number; p50: number; p95: number; min: number; max: number;
}

/** One measured request, as the charts consume it. */
export interface Point {
  id: string; ts: number; model: string; kind: string;
  think: string; cold: number;
  decode_tps: number | null; prefill_tps: number | null;
  ttft_ms: number | null; ttfat_ms: number | null; total_ms: number | null;
  prompt_tokens: number | null; output_tokens: number | null;
  thinking_tokens: number | null; answer_tokens: number | null;
  peak_rss_bytes: number | null; srv_load_ms: number | null;
  itl_stddev_ms: number | null;
}

export interface ModelSummary {
  model: string; n: number;
  decode_tps: Spread | null; prefill_tps: Spread | null;
  ttfat_ms: Spread | null; ttft_ms: Spread | null; total_ms: Spread | null;
  peak_rss_bytes: Spread | null;
  cold: { n: number; load_ms: Spread | null };
  warm: { n: number; load_ms: Spread | null };
  thinking: { thinking_tokens: number; answer_tokens: number; ratio: number | null };
  first_ts: number | null; last_ts: number | null;
}

export interface StatsSummary {
  models: ModelSummary[];
  excluded: Record<string, number>;
  totals: { requests: number; included: number; failed: number };
}

export interface ItemVerdict {
  model: string; score: number; passed: boolean;
  pred: string | null; gold: string | null; request_id: string;
}

export interface DisagreementItem {
  item_id: string; task: string; run_id: string;
  spread: number; models: ItemVerdict[];
}

export interface Disagreements {
  disagreements: DisagreementItem[];
  all_failed: DisagreementItem[];
  n_items: number; n_disagreements: number; n_all_failed: number;
}

export interface ThinkingPoint {
  cell_key: string; model: string; think: string; n: number;
  accuracy: number; thinking_tokens: number; answer_tokens: number;
  think_ratio: number; overthink_rate: number;
  ttft_ms: number | null; ttfat_ms: number | null; total_ms: number | null;
}

export interface DeterminismCell {
  cell_key: string; model: string; temperature: number; think: string;
  n_items: number; samples_per_item: number;
  mean_duplicate_rate: number; fully_deterministic_items: number;
  items: Record<string, {
    n: number; distinct: number; exact_duplicate_rate: number;
    length_mean: number; length_stdev: number;
    score_mean: number | null; score_stdev: number | null;
    fully_deterministic: boolean;
  }>;
}

export interface SessionRow {
  id: string; ts: number; title: string; model: string; messages: number;
}

export interface SessionMessage {
  id: string; session_id: string; ts: number; idx: number;
  role: "user" | "assistant";
  content: string | null; thinking: string | null;
  request_id: string | null; error: string | null;
  ttft_ms: number | null; ttfat_ms: number | null; total_ms: number | null;
  decode_tps: number | null; prefill_tps: number | null;
  prompt_tokens: number | null; output_tokens: number | null;
  thinking_tokens: number | null; answer_tokens: number | null;
  itl_stddev_ms: number | null; srv_load_ms: number | null;
  cold: number | null; model: string | null; think: string | null;
}

export interface SuiteInfo {
  name: string; items: number; verified: number; tasks: string[];
  hash: string; categories: string[]; unanswerable: number;
}

export interface CellResult {
  cell_key: string; label: string; model: string;
  axes: Record<string, unknown>;
  n: number; score: number; ci_lo: number; ci_hi: number; pass_rate: number;
  tasks: Record<string, Record<string, number>>;
  tied_with_previous?: boolean;
}

export interface RunRow {
  id: string; ts: number; finished_ts: number | null;
  suite: string; suite_hash: string; preset: string; label: string;
  status: "running" | "done" | "aborted" | "failed";
  planned_units: number; done_units: number;
  models: string[]; elapsed_s: number | null;
  best_score: number | null; n_cells: number;
}

export interface RunDetail extends Omit<RunRow, "best_score" | "n_cells"> {
  cells: CellResult[];
  plan: Record<string, unknown>;
  abort_reason: string | null;
  run_seed: number;
  env_hash: string;
}

export interface ResultRow {
  id: string; item_id: string; task: string; model: string; cell_key: string;
  score: number; passed: number; pred_label: string | null;
  gold_label: string | null; detail_json: string; request_id: string;
}

async function json<T>(path: string, init?: RequestInit): Promise<T> {
  if (isDemo) {
    const method = (init?.method ?? "GET").toUpperCase();
    // Reads come from the fixture; anything that would mutate is refused by
    // name, so the banner says what it would have done rather than "failed".
    if (method !== "GET") demoRefuse(describeWrite(method, path));
    return demoGet<T>(path);
  }
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
  sessions: () => json<SessionRow[]>("/sessions"),
  session: (id: string) => json<{ id: string; messages: SessionMessage[] }>(`/sessions/${id}`),
  renameSession: (id: string, title: string) =>
    json<{ title: string }>(`/sessions/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title }),
    }),
  deleteSession: (id: string) =>
    json<{ measurements_kept: number }>(`/sessions/${id}`, { method: "DELETE" }),
  statsSummary: () => json<StatsSummary>("/stats/summary"),
  disagreements: (runId = "", limit = 40) =>
    json<Disagreements>(
      `/stats/disagreements?limit=${limit}${runId ? `&run_id=${runId}` : ""}`,
    ),
  timeline: (limit = 1000) =>
    json<{ points: Point[]; models: string[] }>(`/stats/timeline?limit=${limit}`),
  suites: () => json<SuiteInfo[]>("/suites"),
  suiteFromSession: (sessionId: string, name: string, task = "contains") =>
    json<{ name: string; items: number; path: string }>(
      `/suites/from-session/${sessionId}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, task }),
      },
    ),
  runs: (limit = 50) => json<RunRow[]>(`/runs?limit=${limit}`),
  run: (id: string) => json<RunDetail>(`/runs/${id}`),
  thinking: (id: string) =>
    json<{ run_id: string; points: ThinkingPoint[]; note: string | null }>(
      `/runs/${id}/thinking`,
    ),
  determinism: (id: string) =>
    json<{ run_id: string; cells: DeterminismCell[]; note: string | null }>(
      `/runs/${id}/determinism`,
    ),
  runResults: (id: string, failedOnly = false) =>
    json<ResultRow[]>(`/runs/${id}/results?failed_only=${failedOnly}`),
  startRun: (body: {
    models: string[]; suites: string[]; repeats: number;
    think_levels: string[]; temperatures?: number[]; label: string;
  }) =>
    json<{ run_id: string; total_requests: number; cells: number }>("/bench/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  deleteRun: (id: string) =>
    json<{ deleted: string; requests: number; results: number; cells: number }>(
      `/runs/${id}`,
      { method: "DELETE" },
    ),
  abortRun: (id: string) =>
    json<{ aborting: string }>(`/bench/runs/${id}/abort`, { method: "POST" }),
  activeRun: () => json<{ run_id: string } | null>("/bench/active"),
  modelHistory: (n: string, limit = 100) =>
    json<{ model: string; points: Point[]; summary: ModelSummary }>(
      `/models/${n}/history?limit=${limit}`,
    ),
  request: (id: string) => json<Record<string, unknown>>(`/requests/${id}`),
};


/** Human-readable name for a write, used in demo-mode refusals. */
function describeWrite(method: string, path: string): string {
  if (path.startsWith("/models/pull")) return "Pulling a model";
  if (path.includes("/warm")) return "Loading a model into memory";
  if (path.includes("/unload")) return "Unloading a model";
  if (method === "DELETE" && path.startsWith("/models")) return "Deleting a model";
  if (method === "DELETE" && path.startsWith("/runs")) return "Deleting a run";
  if (method === "DELETE" && path.startsWith("/sessions")) return "Deleting a conversation";
  if (path.startsWith("/bench/runs")) return "Starting a benchmark";
  if (path.startsWith("/suites/from-session")) return "Saving a conversation as a suite";
  if (path === "/chat") return "Chatting";
  return "That action";
}

export type SSEHandler = (event: string, data: any) => void;

/** Parse an SSE body, dispatching each frame.
 *  Shared by the POST and GET helpers so the framing is parsed one way only. */
async function readSSE(res: Response, onEvent: SSEHandler): Promise<void> {
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

async function failure(res: Response): Promise<Error> {
  let detail: any;
  try { detail = (await res.json()).detail; } catch { detail = null; }
  return Object.assign(new Error(`${res.status}`), { status: res.status, detail });
}

/** POST that returns an SSE stream. EventSource cannot POST, so the framing is
 *  parsed here — which is needed anyway to stamp arrival times. */
export async function streamPost(
  path: string,
  body: unknown,
  onEvent: SSEHandler,
  signal?: AbortSignal,
): Promise<void> {
  if (isDemo) demoRefuse(describeWrite("POST", path));
  const res = await fetch(`/api${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok) throw await failure(res);
  await readSSE(res, onEvent);
}

/** GET that returns an SSE stream. Used for attaching to a run already in
 *  progress, which is a read rather than an action. */
export async function streamGet(
  path: string,
  onEvent: SSEHandler,
  signal?: AbortSignal,
): Promise<void> {
  if (isDemo) demoRefuse("Watching a live benchmark");
  const res = await fetch(`/api${path}`, { signal });
  if (!res.ok) throw await failure(res);
  await readSSE(res, onEvent);
}

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
