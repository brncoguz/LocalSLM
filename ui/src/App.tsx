import { useCallback, useEffect, useState } from "react";
import { Activity, BarChart3, Boxes, FlaskConical, MessageSquare } from "lucide-react";
import { api, type Health, type ModelsResponse } from "@/lib/api";
import { MachineStrip } from "@/components/MachineStrip";
import { DemoBanner } from "@/components/DemoBanner";
import { Chat } from "@/pages/Chat";
import { Models } from "@/pages/Models";
import { Stats } from "@/pages/Stats";
import { Bench } from "@/pages/Bench";
import { cn } from "@/lib/utils";
import { isDemo } from "@/lib/demo";

type Tab = "chat" | "models" | "bench" | "stats";

const TABS: { id: Tab; label: string; icon: typeof MessageSquare }[] = [
  { id: "chat", label: "Chat", icon: MessageSquare },
  { id: "models", label: "Models", icon: Boxes },
  { id: "bench", label: "Bench", icon: FlaskConical },
  { id: "stats", label: "Stats", icon: BarChart3 },
];

export default function App() {
  const [tab, setTab] = useState<Tab>("chat");
  const [health, setHealth] = useState<Health | null>(null);
  const [models, setModels] = useState<ModelsResponse | null>(null);

  const refresh = useCallback(async () => {
    try {
      const [h, m] = await Promise.all([api.health(), api.models()]);
      setHealth(h);
      setModels(m);
    } catch {
      setHealth(null);
    }
  }, []);

  useEffect(() => {
    void refresh();
    // Nothing changes in demo mode — the fixture is static, so polling would
    // only burn cycles re-reading the same object.
    if (isDemo) return;
    const t = setInterval(refresh, 5000);
    return () => clearInterval(t);
  }, [refresh]);

  return (
    <div className="flex h-screen flex-col">
      <header className="flex items-center gap-4 border-b border-border px-4 py-2">
        <div className="flex items-center gap-2">
          <Activity size={16} className="text-accent" />
          <span className="text-sm font-semibold tracking-tight">LocalSLM</span>
        </div>
        <nav className="flex gap-0.5">
          {TABS.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={cn(
                "flex items-center gap-1.5 rounded-lg px-2.5 py-1 text-[13px] transition-colors",
                tab === t.id ? "bg-surface-2 text-fg" : "text-faint hover:text-muted",
              )}
            >
              <t.icon size={13} />
              {t.label}
            </button>
          ))}
        </nav>
      </header>

      <DemoBanner />
      <MachineStrip health={health} />

      <main className="min-h-0 flex-1">
        {!health?.runtime.ok ? (
          <OllamaDown endpoint={health?.runtime.endpoint} />
        ) : tab === "chat" ? (
          <Chat models={models?.installed ?? []} onMeasured={refresh} />
        ) : tab === "models" ? (
          <Models data={models} refresh={refresh} />
        ) : tab === "bench" ? (
          <Bench models={models?.installed ?? []} />
        ) : (
          <Stats />
        )}
      </main>
    </div>
  );
}

function OllamaDown({ endpoint }: { endpoint?: string }) {
  return (
    <div className="flex h-full items-center justify-center">
      <div className="max-w-sm text-center">
        <p className="text-sm font-medium">Ollama isn't reachable</p>
        <p className="mt-1.5 text-[12px] leading-relaxed text-faint">
          Nothing here works without it. Start it, and this screen recovers on its own.
        </p>
        <code className="mt-3 inline-block rounded bg-surface-2 px-2.5 py-1.5 text-[12px] text-muted">
          ollama serve
        </code>
        {endpoint && <p className="mt-2 text-[11px] text-faint">expected at {endpoint}</p>}
      </div>
    </div>
  );
}
