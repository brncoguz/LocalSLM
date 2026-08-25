import { ExternalLink, Info } from "lucide-react";
import { isDemo } from "@/lib/demo";

/** Says plainly that the numbers are recorded.
 *
 *  A dashboard that looks live while serving frozen data is the same category
 *  of dishonesty this project spends its effort avoiding elsewhere, so the
 *  notice is permanent rather than dismissible. */
export function DemoBanner() {
  if (!isDemo) return null;
  return (
    <div className="flex items-center gap-2 border-b border-accent/25 bg-accent/8 px-4 py-1.5 text-[11px] text-accent">
      <Info size={12} className="shrink-0" />
      <span className="text-muted">
        <span className="font-medium text-accent">Recorded demo.</span> Real
        measurements from a MacBook Pro M1 Pro (16 GB), captured earlier — not a live
        instance. Chatting, pulling models and running benchmarks need a local Ollama.
      </span>
      <a
        href="https://github.com/brncoguz/LocalSLM"
        target="_blank"
        rel="noreferrer"
        className="ml-auto flex shrink-0 items-center gap-1 hover:underline"
      >
        <ExternalLink size={11} /> Run it yourself
      </a>
    </div>
  );
}
