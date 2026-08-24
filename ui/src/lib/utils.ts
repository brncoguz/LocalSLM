import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/** Stable colour identity for a model, keyed by name.
 *  The same model is the same hue everywhere in the app. */
const PALETTE = [
  "var(--color-m1)", "var(--color-m2)", "var(--color-m3)", "var(--color-m4)",
  "var(--color-m5)", "var(--color-m6)", "var(--color-m7)", "var(--color-m8)",
];

export function modelColor(name: string): string {
  let h = 0;
  for (let i = 0; i < name.length; i++) h = (h * 31 + name.charCodeAt(i)) >>> 0;
  return PALETTE[h % PALETTE.length];
}

export const fmt = {
  /** Milliseconds, scaled so the eye can compare at a glance. */
  ms(v: number | null | undefined): string {
    if (v == null) return "—";
    if (v < 1000) return `${Math.round(v)} ms`;
    return `${(v / 1000).toFixed(v < 10000 ? 2 : 1)} s`;
  },
  tps(v: number | null | undefined): string {
    return v == null ? "—" : `${v.toFixed(1)}`;
  },
  bytes(v: number | null | undefined): string {
    if (v == null) return "—";
    if (v < 1024 ** 3) return `${(v / 1024 ** 2).toFixed(0)} MB`;
    return `${(v / 1024 ** 3).toFixed(2)} GB`;
  },
  gb(v: number | null | undefined): string {
    return v == null ? "—" : `${(v / 1e9).toFixed(1)} GB`;
  },
  pct(v: number | null | undefined): string {
    return v == null ? "—" : `${Math.round(v * 100)}%`;
  },
  int(v: number | null | undefined): string {
    return v == null ? "—" : v.toLocaleString();
  },
};
