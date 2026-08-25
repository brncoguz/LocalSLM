"""How much the judge is worth.

An LLM judge is only useful if you know how often it is right, and the one
place that can be checked is where a deterministic grader already has an
answer. So the judge is pointed at items that were graded by exact match,
numeric tolerance or schema validation, asked the same question reference-free,
and its verdict is compared with the grader's.

The headline is **Cohen's kappa, not raw agreement**. On these suites most
items pass, so a judge that answered "pass" to everything would score 85-90%
agreement while knowing nothing. Kappa subtracts the agreement you would get by
chance and is the difference between measuring a judge and flattering one — the
same reason `classify` reports macro-F1 rather than accuracy.

The two ways of being wrong are reported separately because they are not
equally bad. A judge that endorses a wrong answer (`false_pass_rate`) quietly
inflates every score built on it. A judge that rejects a correct one is merely
pessimistic, and shows up as a suspiciously hard suite.
"""

from __future__ import annotations

import json
import time
from sqlite3 import Connection
from typing import Any

from localslm.bench.judge import Judge
from localslm.bench.suites import Suite, TaskItem, load
from localslm.db import insert, new_id, query

# Nothing to calibrate against: the rubric grader *is* the judge.
DETERMINISTIC_ONLY = "res.task != 'rubric'"


def _suite_items(run: dict) -> tuple[dict[str, TaskItem], list[str]]:
    """item_id -> item for every suite the run used, plus suites that vanished.

    Suites are re-read from disk because runs record their plan, not their
    items. That means a suite edited since the run would be judged against its
    current text, so items no longer present are reported rather than skipped
    silently.
    """
    items: dict[str, TaskItem] = {}
    missing: list[str] = []
    for name in (run.get("suite") or "").split(","):
        name = name.strip()
        if not name:
            continue
        try:
            suite: Suite = load(name)
        except FileNotFoundError:
            missing.append(name)
            continue
        for item in suite.items:
            items[item.id] = item
    return items, missing


def pending_rows(conn: Connection, run_id: str, limit: int = 0) -> list[dict]:
    """Graded, non-excluded results for this run that no judge has seen yet.

    Excluded rows are skipped for the same reason every other consumer skips
    them, and re-judging is avoided so the command can be resumed after an
    interrupt without doubling the verdicts.
    """
    sql = f"""
        SELECT res.id, res.request_id, res.item_id, res.task, res.model,
               res.score, res.passed, req.output_text
        FROM results res
        JOIN requests req ON req.id = res.request_id
        WHERE res.run_id = ? AND req.excluded = 0 AND {DETERMINISTIC_ONLY}
          AND res.request_id NOT IN (
              SELECT request_id FROM judge_verdicts WHERE run_id = ?
          )
        ORDER BY res.id
    """
    if limit:
        sql += f" LIMIT {int(limit)}"
    return [dict(r) for r in query(conn, sql, (run_id, run_id))]


async def judge_run(
    conn: Connection,
    judge: Judge,
    run_id: str,
    *,
    limit: int = 0,
    on_verdict: Any = None,
) -> dict[str, Any]:
    """Score a completed run's answers with the judge, storing every verdict.

    Verdicts are written as they arrive rather than batched at the end: judging
    a large run takes minutes on a local 9B, and an interrupt should keep the
    work already done.
    """
    runs = query(conn, "SELECT id, suite FROM runs WHERE id = ?", (run_id,))
    if not runs:
        raise KeyError(f"unknown run {run_id!r}")
    items, missing_suites = _suite_items(dict(runs[0]))

    rows = pending_rows(conn, run_id, limit)
    judged = 0
    unresolved: list[str] = []

    for row in rows:
        item = items.get(row["item_id"])
        if item is None:
            # The suite changed since the run. Judging against a different
            # question than the model answered would be worse than not judging.
            unresolved.append(row["item_id"])
            continue

        verdict = await judge.score(
            item,
            row["output_text"] or "",
            reference=None,
            parent_id=row["request_id"],
            run_id=run_id,
        )
        insert(
            conn,
            "judge_verdicts",
            {
                "id": new_id(),
                "ts": time.time(),
                "run_id": run_id,
                "request_id": row["request_id"],
                "judge_request_id": verdict.judge_request_id,
                "item_id": row["item_id"],
                "task": row["task"],
                "model": row["model"],
                "judge_model": verdict.judge_model,
                "backend": verdict.backend,
                "reference_free": 1,
                "rating": verdict.rating,
                "score": verdict.score,
                "passed": None if verdict.passed is None else int(verdict.passed),
                "parse_ok": int(verdict.parse_ok),
                "rationale": verdict.rationale[:2000],
                "raw": verdict.raw,
                "detail_json": None,
                "cost_usd": verdict.cost_usd,
            },
        )
        judged += 1
        if on_verdict:
            on_verdict(row, verdict)

    return {
        "run_id": run_id,
        "judged": judged,
        "pending": len(rows),
        "unresolved_items": unresolved,
        "missing_suites": missing_suites,
    }


def _kappa(
    both_pass: int, judge_only: int, grader_only: int, both_fail: int
) -> float | None:
    """Cohen's kappa for two binary raters.

    None when it is undefined — which happens exactly when one rater said the
    same thing about everything, and there is then no agreement to measure
    beyond chance. Reporting 0.0 there would read as "the judge is worthless"
    rather than "this suite cannot tell us".
    """
    n = both_pass + judge_only + grader_only + both_fail
    if n == 0:
        return None
    observed = (both_pass + both_fail) / n
    judge_pass = (both_pass + judge_only) / n
    grader_pass = (both_pass + grader_only) / n
    expected = judge_pass * grader_pass + (1 - judge_pass) * (1 - grader_pass)
    if expected >= 1.0:
        return None
    return (observed - expected) / (1 - expected)


def verdicts(conn: Connection, run_id: str, only: str = "all", limit: int = 60) -> dict:
    """Every judgement with the evidence behind it.

    The agreement report says *how often* the judge was right. This says *what
    it was doing*, which is the only way to tell a judge that reasons from one
    that pattern-matches — and those score identically until you read them.

    Each row carries the answer that was graded, the grader's verdict, and the
    judge's rating, reason and raw reply. The judge's own prompt is included so
    a suspicious verdict can be traced to what it was actually shown rather
    than what we assume it was shown.
    """
    where = ["v.run_id = ?"]
    args: list[Any] = [run_id]
    if only == "disagreements":
        where.append("v.parse_ok = 1 AND res.passed IS NOT NULL AND v.passed != res.passed")
    elif only == "parse_failures":
        where.append("v.parse_ok = 0")

    rows = query(
        conn,
        f"""SELECT v.item_id, v.task, v.model, v.rating, v.score, v.passed AS judge_passed,
                   v.parse_ok, v.rationale, v.raw, v.reference_free, v.judge_model,
                   res.passed AS grader_passed, res.score AS grader_score,
                   res.gold_label, res.pred_label,
                   areq.output_text AS answer, areq.prompt_json AS answer_prompt,
                   jreq.prompt_json AS judge_prompt, jreq.total_ms AS judge_ms,
                   jreq.output_tokens AS judge_tokens
            FROM judge_verdicts v
            LEFT JOIN results res ON res.request_id = v.request_id
            LEFT JOIN requests areq ON areq.id = v.request_id
            LEFT JOIN requests jreq ON jreq.id = v.judge_request_id
            WHERE {" AND ".join(where)}
            ORDER BY v.ts
            LIMIT ?""",
        (*args, limit),
    )

    out = []
    for r in rows:
        judge_prompt = ""
        if r["judge_prompt"]:
            try:
                # Stored as [[role, content], ...]; the user turn is the one
                # carrying the rubric, the question and the fenced answer.
                turns = json.loads(r["judge_prompt"])
                judge_prompt = "\n\n".join(f"[{role}]\n{text}" for role, text in turns)
            except (ValueError, TypeError):
                judge_prompt = ""
        out.append(
            {
                "item_id": r["item_id"],
                "task": r["task"],
                "model": r["model"],
                "question": _first_user_turn(r["answer_prompt"]),
                "answer": (r["answer"] or "")[:1000],
                "grader_passed": (
                    None if r["grader_passed"] is None else bool(r["grader_passed"])
                ),
                "grader_score": r["grader_score"],
                "gold_label": r["gold_label"],
                "judge_model": r["judge_model"],
                "rating": r["rating"],
                "judge_passed": None
                if r["judge_passed"] is None
                else bool(r["judge_passed"]),
                "parse_ok": bool(r["parse_ok"]),
                "reason": r["rationale"] or "",
                "raw": (r["raw"] or "")[:1000],
                "judge_prompt": judge_prompt[:4000],
                "judge_ms": r["judge_ms"],
                "judge_tokens": r["judge_tokens"],
                "agreed": (
                    None
                    if (not r["parse_ok"] or r["grader_passed"] is None)
                    else bool(r["judge_passed"]) == bool(r["grader_passed"])
                ),
            }
        )
    return {"run_id": run_id, "only": only, "n": len(out), "verdicts": out}


def _first_user_turn(prompt_json: str | None) -> str:
    """The question as the model saw it, for showing beside its answer."""
    if not prompt_json:
        return ""
    try:
        turns = json.loads(prompt_json)
    except (ValueError, TypeError):
        return ""
    for role, text in turns:
        if role == "user":
            return text[:1000]
    return ""


def rating_histogram(conn: Connection, run_id: str) -> dict[str, int]:
    """How the judge spent the scale.

    The single most diagnostic view of a judge. One that rates everything 5 is
    not grading, and it can still post a respectable agreement rate on a suite
    most models pass — the histogram makes that obvious at a glance where the
    headline number hides it.
    """
    rows = query(
        conn,
        "SELECT rating, COUNT(*) n FROM judge_verdicts "
        "WHERE run_id = ? AND parse_ok = 1 AND rating IS NOT NULL GROUP BY rating",
        (run_id,),
    )
    return {str(r["rating"]): r["n"] for r in rows}


def agreement(conn: Connection, run_id: str) -> dict[str, Any]:
    """Compare stored verdicts against the deterministic grades for one run."""
    rows = query(
        conn,
        f"""SELECT v.passed AS judge_passed, v.parse_ok, v.rating, v.judge_model,
                   v.task, v.model, v.item_id, v.rationale,
                   res.passed AS grader_passed
            FROM judge_verdicts v
            JOIN results res ON res.request_id = v.request_id
            WHERE v.run_id = ? AND {DETERMINISTIC_ONLY}""",
        (run_id,),
    )
    if not rows:
        return {
            "run_id": run_id,
            "n": 0,
            "note": "No verdicts stored for this run. Run `localslm judge <run_id>` first.",
        }

    usable = [r for r in rows if r["parse_ok"] and r["judge_passed"] is not None]
    both_pass = sum(1 for r in usable if r["judge_passed"] and r["grader_passed"])
    judge_only = sum(1 for r in usable if r["judge_passed"] and not r["grader_passed"])
    grader_only = sum(1 for r in usable if not r["judge_passed"] and r["grader_passed"])
    both_fail = sum(1 for r in usable if not r["judge_passed"] and not r["grader_passed"])
    n = len(usable)

    disagreements = [
        {
            "item_id": r["item_id"],
            "model": r["model"],
            "task": r["task"],
            "grader": "pass" if r["grader_passed"] else "fail",
            "judge": "pass" if r["judge_passed"] else "fail",
            "rating": r["rating"],
            "reason": r["rationale"],
        }
        for r in usable
        if bool(r["judge_passed"]) != bool(r["grader_passed"])
    ]

    return {
        "run_id": run_id,
        "judge_model": rows[0]["judge_model"],
        "n": n,
        # How the judge spent the scale. A judge that only ever says 5 is not
        # grading, and the headline agreement rate will not show that.
        "rating_histogram": rating_histogram(conn, run_id),
        "n_verdicts": len(rows),
        "judge_parse_failure_rate": 1 - (n / len(rows)) if rows else 0.0,
        "agreement_rate": (both_pass + both_fail) / n if n else None,
        # The number to actually read. Raw agreement is inflated by these
        # suites being mostly passable.
        "kappa": _kappa(both_pass, judge_only, grader_only, both_fail),
        # A judge that says "pass" to everything scores well on agreement and
        # badly here, which is the whole point of printing both.
        "judge_pass_rate": (both_pass + judge_only) / n if n else None,
        "grader_pass_rate": (both_pass + grader_only) / n if n else None,
        "false_pass_rate": judge_only / n if n else None,
        "false_fail_rate": grader_only / n if n else None,
        "confusion": {
            "both_pass": both_pass,
            "judge_pass_grader_fail": judge_only,
            "judge_fail_grader_pass": grader_only,
            "both_fail": both_fail,
        },
        "disagreements": disagreements[:40],
    }
