import { useEffect, useState } from "react";
import { ChevronRight } from "lucide-react";
import { api, type JudgeVerdict, type VerdictFilter } from "@/lib/api";
import { cn } from "@/lib/utils";

/** What the judge actually did, one item at a time.
 *
 *  The agreement report answers "how often was it right". This answers "was it
 *  right for the right reason", which is a different question and the one that
 *  decides whether a judge can be trusted on open-ended items where no grader
 *  exists to check it. A judge that reasons and a judge that pattern-matches
 *  score identically until someone reads the verdicts.
 *
 *  Each row expands to the full evidence — the question, the answer graded,
 *  both verdicts, and the judge's raw reply. The prompt the judge was given is
 *  available too, because a surprising verdict is often a surprising prompt. */
export function JudgeVerdicts({ runId }: { runId: string }) {
  const [filter, setFilter] = useState<VerdictFilter>("disagreements");
  const [rows, setRows] = useState<JudgeVerdict[] | null>(null);
  const [open, setOpen] = useState<string | null>(null);

  useEffect(() => {
    setRows(null);
    void api
      .judgeVerdicts(runId, filter)
      .then((d) => setRows(d.verdicts))
      .catch(() => setRows([]));
  }, [runId, filter]);

  return (
    <div>
      <div className="mb-2 flex flex-wrap items-center gap-1">
        <p className="mr-auto text-[11px] text-faint">
          what the judge did — read these before trusting the number above
        </p>
        {FILTERS.map(([key, label]) => (
          <button
            key={key}
            onClick={() => setFilter(key)}
            className={cn(
              "rounded-md px-1.5 py-0.5 text-[11px] transition-colors",
              filter === key
                ? "bg-accent/15 text-accent"
                : "text-faint hover:bg-surface-2 hover:text-muted",
            )}
          >
            {label}
          </button>
        ))}
      </div>

      {rows === null ? (
        <p className="text-[11px] text-faint">Loading…</p>
      ) : !rows.length ? (
        <p className="text-[11px] text-faint">
          {filter === "disagreements"
            ? "The judge agreed with the grader on every item it could read."
            : filter === "parse_failures"
              ? "Every verdict parsed cleanly."
              : "No verdicts yet."}
        </p>
      ) : (
        <ul className="space-y-1">
          {rows.map((v) => {
            const id = `${v.item_id}-${v.model}`;
            return (
              <li key={id} className="rounded-md border border-border bg-surface-2">
                <button
                  onClick={() => setOpen(open === id ? null : id)}
                  className="flex w-full items-center gap-2 px-2 py-1.5 text-left text-[11px]"
                >
                  <ChevronRight
                    size={11}
                    className={cn(
                      "shrink-0 text-faint transition-transform",
                      open === id && "rotate-90",
                    )}
                  />
                  <span className="tnum font-medium">{v.item_id}</span>
                  <span className="tnum text-faint">{v.model}</span>
                  <Rating rating={v.rating} parseOk={v.parse_ok} />
                  <Outcome label="grader" passed={v.grader_passed} />
                  <Outcome label="judge" passed={v.judge_passed} />
                  <span className="ml-auto shrink-0 text-faint">
                    {v.agreed === null ? "—" : v.agreed ? "agreed" : "differed"}
                  </span>
                </button>

                {open === id && (
                  <div className="space-y-2 border-t border-border px-2.5 py-2 text-[11px]">
                    <Field label="question">{v.question || "—"}</Field>
                    <Field label={`answer from ${v.model}`}>{v.answer || "(empty)"}</Field>
                    {v.gold_label && <Field label="expected">{v.gold_label}</Field>}
                    <Field label={`${v.judge_model} said`}>
                      {v.parse_ok ? (
                        <>
                          <span className="font-medium">{v.rating}/5</span>
                          {v.reason ? ` — ${v.reason}` : ""}
                        </>
                      ) : (
                        <span className="text-bad">
                          nothing usable — this is a fact about the judge, not the model
                        </span>
                      )}
                    </Field>
                    <Field label="raw reply">
                      <code className="break-all">{v.raw || "(empty)"}</code>
                    </Field>
                    {v.judge_ms != null && (
                      <p className="tnum text-faint">
                        cost {(v.judge_ms / 1000).toFixed(1)}s
                        {v.judge_tokens != null && ` · ${v.judge_tokens} tokens`}
                      </p>
                    )}
                    {v.judge_prompt && (
                      <details>
                        <summary className="cursor-pointer text-faint">
                          what the judge was shown
                        </summary>
                        <pre className="mt-1 max-h-64 overflow-auto whitespace-pre-wrap rounded bg-surface p-2 text-[10px] leading-relaxed text-muted">
                          {v.judge_prompt}
                        </pre>
                      </details>
                    )}
                  </div>
                )}
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}

const FILTERS: [VerdictFilter, string][] = [
  ["disagreements", "disagreements"],
  ["parse_failures", "unreadable"],
  ["all", "all"],
];

function Rating({ rating, parseOk }: { rating: number | null; parseOk: boolean }) {
  if (!parseOk) return <span className="shrink-0 text-bad">unreadable</span>;
  return (
    <span className="tnum shrink-0 text-faint" title="the judge's 1-5 rating">
      {rating}/5
    </span>
  );
}

function Outcome({ label, passed }: { label: string; passed: boolean | null }) {
  if (passed === null) return null;
  return (
    <span className={cn("shrink-0", passed ? "text-good" : "text-bad")}>
      {label}: {passed ? "pass" : "fail"}
    </span>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-wide text-faint">{label}</div>
      <div className="mt-0.5 whitespace-pre-wrap text-muted">{children}</div>
    </div>
  );
}
