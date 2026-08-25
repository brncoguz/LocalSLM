/** Static demo mode.
 *
 *  The published demo has to run without a backend: nobody reviewing this
 *  repository is going to install Ollama and pull several gigabytes of weights
 *  to look at a dashboard. So every read is answered from a fixture captured
 *  from the real API (`make demo-data`), and every write is refused with an
 *  explanation rather than silently doing nothing.
 */
import fixture from "@/fixtures/demo.json";

export const isDemo = import.meta.env.VITE_DEMO === "1";

const snapshot = fixture as Record<string, unknown>;

/** Raised for anything that would need a live engine. The UI already renders
 *  `detail.message` for failures, so these surface as normal banners. */
export class DemoUnavailable extends Error {
  detail: { error: string; message: string };
  constructor(message: string) {
    super(message);
    this.detail = { error: "demo_mode", message };
  }
}

function lookup(path: string): unknown | undefined {
  const full = path.startsWith("/api") ? path : `/api${path}`;
  if (full in snapshot) return snapshot[full];

  // Fall back to ignoring the query string: the fixture was captured with one
  // set of limits, and the app may ask for another. Serving the recorded page
  // is better than a blank screen, and the demo is not paginated anyway.
  const bare = full.split("?")[0];
  for (const key of Object.keys(snapshot)) {
    if (key.split("?")[0] === bare) return snapshot[key];
  }
  return undefined;
}

/** Answer a GET from the fixture. Throws if the path was never captured. */
export function demoGet<T>(path: string): T {
  const hit = lookup(path);
  if (hit === undefined) {
    throw new DemoUnavailable(
      "This view isn't part of the recorded demo. Run the project locally to see it.",
    );
  }
  return hit as T;
}

export function demoHas(path: string): boolean {
  return lookup(path) !== undefined;
}

/** Refuse anything that needs the engine, naming what it would have done. */
export function demoRefuse(what: string): never {
  throw new DemoUnavailable(
    `${what} needs a running Ollama instance. This is a recorded demo — ` +
      `clone the repo and run it locally to do this for real.`,
  );
}
