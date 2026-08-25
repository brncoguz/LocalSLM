import { Cpu, HardDrive, MemoryStick, Plug, BatteryMedium, AlertTriangle } from "lucide-react";
import type { Health } from "@/lib/api";
import { fmt } from "@/lib/utils";

/** Always-visible machine context.
 *  Not decoration: memory headroom and power source change what every number
 *  below them means, so they stay on screen at all times. */
export function MachineStrip({ health }: { health: Health | null }) {
  if (!health) {
    return <div className="h-9 border-b border-border bg-surface" />;
  }
  const { memory, disk, power, host, runtime } = health;
  const pressure = memory.pressure;
  const tone =
    pressure === "critical" ? "text-bad" : pressure === "warn" ? "text-warn" : "text-muted";

  return (
    <div className="flex h-9 items-center gap-4 overflow-x-auto border-b border-border bg-surface px-4 text-[11px] text-muted">
      <span className="flex items-center gap-1.5 font-medium text-fg">
        <Cpu size={12} className="text-accent" />
        {host.chip} · {host.ram_gb} GB
      </span>

      <span className={`flex items-center gap-1.5 tnum ${tone}`}>
        <MemoryStick size={12} />
        {fmt.gb(memory.free_bytes)} free
        {memory.runner_rss_bytes ? ` · runners ${fmt.gb(memory.runner_rss_bytes)}` : ""}
        {pressure !== "normal" && (
          <span className="flex items-center gap-1">
            <AlertTriangle size={11} />
            {pressure}
          </span>
        )}
      </span>

      <span className="flex items-center gap-1.5 tnum">
        <HardDrive size={12} />
        {fmt.gb(disk.free_bytes)} free
      </span>

      {/* Three states, not two: an unreadable `pmset` must not render as
          "battery", or every measurement taken on an unknown source would be
          silently attributed to the throttled one. */}
      <span
        className={`flex items-center gap-1.5 tnum ${power.on_ac === true ? "" : "text-warn"}`}
        title={
          power.on_ac === null
            ? "Power source could not be read — these measurements are recorded as unknown."
            : undefined
        }
      >
        {power.on_ac === true ? <Plug size={12} /> : <BatteryMedium size={12} />}
        {power.on_ac === true
          ? "AC"
          : power.on_ac === false
            ? `battery ${power.battery_pct ?? "?"}%`
            : "power unknown"}
      </span>

      <span className="ml-auto flex items-center gap-3 shrink-0">
        {health.lease.busy && health.lease.exclusive && (
          <span className="rounded bg-warn/12 px-1.5 py-0.5 text-warn">benchmark running</span>
        )}
        <span className="tnum text-faint">{health.db.requests} measured</span>
        <span className={`flex items-center gap-1 ${runtime.ok ? "text-good" : "text-bad"}`}>
          <span className={`h-1.5 w-1.5 rounded-full ${runtime.ok ? "bg-good" : "bg-bad"}`} />
          ollama {runtime.version ?? "offline"}
        </span>
      </span>
    </div>
  );
}
