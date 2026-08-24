import { useState } from "react";
import { Check, MessageSquarePlus, Pencil, Trash2, X } from "lucide-react";
import type { SessionRow } from "@/lib/api";
import { Button } from "@/components/ui/primitives";
import { cn, modelColor } from "@/lib/utils";

/** Conversation history.
 *
 *  Deleting a conversation removes the scrollback but keeps its measurements —
 *  the confirm says so, because "delete" that quietly shrinks your dataset
 *  would be the wrong contract for a measurement tool. */
export function SessionSidebar({
  sessions, activeId, onSelect, onNew, onRename, onDelete,
}: {
  sessions: SessionRow[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onNew: () => void;
  onRename: (id: string, title: string) => void;
  onDelete: (id: string) => void;
}) {
  const [editing, setEditing] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const [confirming, setConfirming] = useState<string | null>(null);

  const groups = groupByDay(sessions);

  return (
    <aside className="flex w-60 shrink-0 flex-col border-r border-border">
      <div className="p-2">
        <Button variant="outline" className="w-full justify-start" size="sm" onClick={onNew}>
          <MessageSquarePlus size={13} /> New chat
        </Button>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto px-2 pb-2">
        {sessions.length === 0 && (
          <p className="px-2 py-6 text-center text-[11px] leading-relaxed text-faint">
            Conversations you start appear here, with the measurements they produced.
          </p>
        )}

        {groups.map(([label, rows]) => (
          <div key={label} className="mb-3">
            <p className="px-2 pb-1 text-[10px] uppercase tracking-wide text-faint">{label}</p>
            {rows.map((s) => {
              const active = s.id === activeId;
              return (
                <div
                  key={s.id}
                  className={cn(
                    "group relative rounded-lg px-2 py-1.5 text-[12px] transition-colors",
                    active ? "bg-surface-2" : "hover:bg-surface",
                  )}
                >
                  {editing === s.id ? (
                    <div className="flex items-center gap-1">
                      <input
                        autoFocus
                        value={draft}
                        onChange={(e) => setDraft(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === "Enter" && draft.trim()) {
                            onRename(s.id, draft.trim());
                            setEditing(null);
                          } else if (e.key === "Escape") setEditing(null);
                        }}
                        className="min-w-0 flex-1 rounded border border-border bg-bg px-1.5 py-0.5 text-[12px] focus:outline-none focus:ring-1 focus:ring-accent/50"
                      />
                      <button
                        className="text-faint hover:text-good"
                        onClick={() => { if (draft.trim()) onRename(s.id, draft.trim()); setEditing(null); }}
                      >
                        <Check size={12} />
                      </button>
                      <button className="text-faint hover:text-fg" onClick={() => setEditing(null)}>
                        <X size={12} />
                      </button>
                    </div>
                  ) : confirming === s.id ? (
                    <div className="text-[11px]">
                      <p className="text-muted">Delete this conversation?</p>
                      <p className="mt-0.5 text-faint">Its measurements are kept.</p>
                      <div className="mt-1.5 flex gap-1">
                        <Button size="sm" variant="danger" className="h-6 px-2 text-[11px]"
                          onClick={() => { onDelete(s.id); setConfirming(null); }}>
                          Delete
                        </Button>
                        <Button size="sm" variant="ghost" className="h-6 px-2 text-[11px]"
                          onClick={() => setConfirming(null)}>
                          Cancel
                        </Button>
                      </div>
                    </div>
                  ) : (
                    <>
                      <button
                        onClick={() => onSelect(s.id)}
                        className="block w-full pr-11 text-left"
                      >
                        <span className={cn("block truncate", active ? "text-fg" : "text-muted")}>
                          {s.title || "Untitled"}
                        </span>
                        <span className="tnum mt-0.5 flex items-center gap-1.5 text-[10px] text-faint">
                          <span
                            className="h-1.5 w-1.5 shrink-0 rounded-full"
                            style={{ background: modelColor(s.model) }}
                          />
                          <span className="truncate">{s.model}</span>
                          <span>· {s.messages}</span>
                        </span>
                      </button>
                      <div className="absolute right-1.5 top-1.5 hidden gap-0.5 group-hover:flex">
                        <button
                          title="Rename"
                          className="rounded p-0.5 text-faint hover:bg-surface-2 hover:text-fg"
                          onClick={() => { setEditing(s.id); setDraft(s.title); }}
                        >
                          <Pencil size={11} />
                        </button>
                        <button
                          title="Delete"
                          className="rounded p-0.5 text-faint hover:bg-surface-2 hover:text-bad"
                          onClick={() => setConfirming(s.id)}
                        >
                          <Trash2 size={11} />
                        </button>
                      </div>
                    </>
                  )}
                </div>
              );
            })}
          </div>
        ))}
      </div>
    </aside>
  );
}

function groupByDay(sessions: SessionRow[]): [string, SessionRow[]][] {
  const today = new Date().toDateString();
  const yesterday = new Date(Date.now() - 86_400_000).toDateString();
  const out = new Map<string, SessionRow[]>();
  for (const s of sessions) {
    const d = new Date(s.ts * 1000).toDateString();
    const label = d === today ? "Today" : d === yesterday ? "Yesterday"
      : new Date(s.ts * 1000).toLocaleDateString(undefined, { month: "short", day: "numeric" });
    (out.get(label) ?? out.set(label, []).get(label)!).push(s);
  }
  return [...out.entries()];
}
