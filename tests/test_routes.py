"""Route contracts.

Deliberately thin. These assert shape rather than values — status codes,
required keys, and above all that an empty database produces an empty-but-valid
response instead of a 500. The correctness of what the numbers *mean* is
asserted in `test_stats.py`, `test_aggregates.py` and `test_thinking_endpoint.py`;
this file is the net that catches a route breaking outright.

A fresh install has no measurements, so every one of these paths runs against
an empty table on someone's first launch. That is the case least likely to be
exercised by hand and most likely to be seen by a stranger.
"""

from __future__ import annotations

import pytest

from localslm.db import insert

EMPTY_OK = [
    "/api/stats/timeline",
    "/api/stats/summary",
    "/api/stats/disagreements",
    "/api/runs",
    "/api/suites",
    "/api/bench/active",
]


@pytest.mark.parametrize("path", EMPTY_OK)
def test_an_empty_database_answers_rather_than_failing(client, path):
    assert client.get(path).status_code == 200


def test_models_lists_the_catalogue_with_nothing_installed(client):
    body = client.get("/api/models").json()

    assert body["installed"] == []
    assert body["catalog"], "the curated catalogue is static, not measured"
    assert all("fits" in entry for entry in body["catalog"]), (
        "each entry says whether it fits the free disk it was checked against"
    )


def test_timeline_returns_points_oldest_first(client, conn):
    """The chart draws left to right. The query is `ORDER BY ts DESC` with a
    limit — newest first so the limit keeps recent data — then reversed."""
    for idx, ts in enumerate((3.0, 1.0, 2.0)):
        insert(
            conn,
            "requests",
            {
                "id": f"r{idx}",
                "ts": ts,
                "kind": "chat",
                "runtime": "fake",
                "model": "m:4b",
                "decode_tps": 100.0,
            },
        )

    points = client.get("/api/stats/timeline").json()["points"]

    assert [p["ts"] for p in points] == [1.0, 2.0, 3.0]


def test_excluded_rows_never_reach_the_timeline(client, conn):
    """`INCLUDED` lives in the SQL precisely so no chart can forget it. If this
    breaks, every chart in the app silently starts averaging swapped rows."""
    for idx, excluded in enumerate((0, 1)):
        insert(
            conn,
            "requests",
            {
                "id": f"r{idx}",
                "ts": 1.0 + idx,
                "kind": "chat",
                "runtime": "fake",
                "model": "m:4b",
                "decode_tps": 100.0,
                "excluded": excluded,
                "exclude_reason": "swapped" if excluded else None,
            },
        )

    points = client.get("/api/stats/timeline").json()["points"]

    assert [p["id"] for p in points] == ["r0"]


class TestRunEndpoints:
    def seed_run(self, conn, run_id="r1"):
        insert(conn, "runs", {"id": run_id, "ts": 1.0, "suite": "math", "status": "done"})
        return run_id

    def test_a_missing_run_is_a_404_not_a_crash(self, client):
        assert client.get("/api/runs/nope").status_code == 404

    @pytest.mark.parametrize("suffix", ["", "/thinking", "/determinism", "/results"])
    def test_a_run_with_no_results_still_answers(self, client, conn, suffix):
        """Aborted runs exist — a run can die on its first unit. Its detail
        pages have to render something rather than 500."""
        run_id = self.seed_run(conn)

        assert client.get(f"/api/runs/{run_id}{suffix}").status_code == 200

    def test_compare_rejects_cells_that_are_not_in_the_run(self, client, conn):
        run_id = self.seed_run(conn)

        r = client.get(f"/api/runs/{run_id}/compare", params={"a": "x", "b": "y"})

        assert r.status_code in (400, 404, 422)


class TestChatSessions:
    """The session lifecycle, and the one guarantee the README makes about it."""

    def seed(self, conn, session_id="s1", request_id="q1"):
        insert(
            conn,
            "chat_sessions",
            {
                "id": session_id,
                "ts": 1.0,
                "updated_ts": 1.0,
                "title": "First conversation",
                "model": "m:4b",
            },
        )
        insert(
            conn,
            "requests",
            {
                "id": request_id,
                "ts": 1.0,
                "kind": "chat",
                "session_id": session_id,
                "runtime": "fake",
                "model": "m:4b",
                "decode_tps": 100.0,
            },
        )
        return session_id

    def test_sessions_are_listed(self, client, conn):
        self.seed(conn)

        body = client.get("/api/sessions").json()

        assert [s["id"] for s in body] == ["s1"]

    def test_a_session_can_be_renamed(self, client, conn):
        session_id = self.seed(conn)

        r = client.patch(f"/api/sessions/{session_id}", json={"title": "Renamed"})

        assert r.status_code == 200
        assert client.get("/api/sessions").json()[0]["title"] == "Renamed"

    def test_deleting_a_session_keeps_its_measurements(self, client, conn):
        """The README states this deliberately: "Deleting a conversation removes
        the transcript but keeps the measurements." The rows outlive the
        transcript because they are data, not chat history — and nothing else
        in the suite checks it."""
        session_id = self.seed(conn)

        assert client.delete(f"/api/sessions/{session_id}").status_code == 200
        assert client.get("/api/sessions").json() == []

        remaining = conn.execute("SELECT COUNT(*) n FROM requests").fetchone()["n"]
        assert remaining == 1

    def test_an_unknown_session_reads_as_empty_rather_than_missing(self, client):
        """Pinning current behaviour, which is not obviously the right one:
        `session_detail` queries messages by id and returns whatever it finds,
        so an unknown id and a deleted one are both an empty conversation
        rather than a 404. Harmless for the sidebar, which only ever asks for
        ids it just listed — but worth stating, because a caller that wanted to
        detect a stale link cannot.
        """
        r = client.get("/api/sessions/nope")

        assert r.status_code == 200
        assert r.json()["messages"] == []


class TestJudgeEndpoints:
    """Starting a judging pass from the UI.

    The failure modes worth covering are the ones a user meets first: a run
    that doesn't exist, a judge model that isn't pulled, and a machine already
    busy. Each has to say which of those happened, because the remedy differs
    and "could not judge" sends people to the wrong place.
    """

    def seed(self, conn, run_id="r1", excluded=0):
        insert(conn, "runs", {"id": run_id, "ts": 1.0, "suite": "math", "status": "done"})
        insert(
            conn,
            "requests",
            {
                "id": "req0",
                "ts": 1.0,
                "kind": "bench",
                "run_id": run_id,
                "runtime": "fake",
                "model": "m:4b",
                "output_text": "4",
                "excluded": excluded,
            },
        )
        insert(
            conn,
            "results",
            {
                "id": "res0",
                "run_id": run_id,
                "request_id": "req0",
                "model": "m:4b",
                "item_id": "m1",
                "task": "numeric",
                "grader": "numeric",
                "score": 1.0,
                "passed": 1,
            },
        )
        return run_id

    def test_an_unknown_run_is_a_404(self, client):
        assert client.post("/api/runs/nope/judge", json={"limit": 5}).status_code == 404

    def test_an_uninstalled_judge_model_says_which_one(self, client, conn):
        """The fake runtime has nothing installed, which is exactly the state a
        fresh checkout is in. Naming the model is the difference between a
        dead end and a one-line fix."""
        run_id = self.seed(conn)

        r = client.post(f"/api/runs/{run_id}/judge", json={"limit": 5})

        assert r.status_code == 404
        body = r.json()["detail"]
        assert body["error"] == "judge_missing"
        assert body["model"]

    def test_a_fully_judged_run_reports_nothing_to_do(self, client, conn):
        """Not an error: re-judging is how the command resumes, so arriving at
        "already done" is a normal outcome and must not read as a failure."""
        run_id = self.seed(conn)
        insert(
            conn,
            "judge_verdicts",
            {
                "id": "v0",
                "ts": 1.0,
                "run_id": run_id,
                "request_id": "req0",
                "judge_model": "j:9b",
                "backend": "local",
                "parse_ok": 1,
                "passed": 1,
            },
        )

        r = client.post(f"/api/runs/{run_id}/judge", json={"limit": 0})

        assert r.status_code == 200
        assert r.json()["pending"] == 0

    def test_streaming_a_run_that_is_not_being_judged_is_a_404(self, client, conn):
        run_id = self.seed(conn)

        assert client.get(f"/api/runs/{run_id}/judge/stream").status_code == 404

    def test_agreement_is_readable_before_anything_is_judged(self, client, conn):
        """The panel calls this on every run it renders, including ones nobody
        has judged. It has to answer rather than 404, or the UI cannot tell an
        unjudged run from a broken endpoint."""
        run_id = self.seed(conn)

        r = client.get(f"/api/runs/{run_id}/judge-agreement")

        assert r.status_code == 200
        assert r.json()["n"] == 0
