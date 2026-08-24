"""The timing maths. This is where the project's credibility lives."""

from localslm.runtimes.base import GenResult, ModelRef, ServerTiming

REF = ModelRef(runtime="fake", name="m")


def test_server_timing_converts_ns_to_ms_and_derives_rates():
    st = ServerTiming.from_done(
        {
            "total_duration": 2_000_000_000,
            "load_duration": 300_000_000,
            "prompt_eval_count": 100,
            "prompt_eval_duration": 500_000_000,
            "eval_count": 200,
            "eval_duration": 1_000_000_000,
        }
    )
    assert st.total_ms == 2000
    assert st.load_ms == 300
    assert st.prefill_tps == 200.0  # 100 tokens / 0.5 s
    assert st.decode_tps == 200.0  # 200 tokens / 1.0 s


def test_rates_are_none_rather_than_dividing_by_zero():
    st = ServerTiming.from_done({"eval_count": 5})
    assert st.decode_tps is None
    assert st.prefill_tps is None


def test_itl_percentiles_and_jitter():
    r = GenResult(model=REF, inter_token_ms=[10, 10, 10, 10, 100])
    assert r.itl_p50_ms == 10
    assert r.itl_p95_ms == 100
    assert r.itl_stddev_ms > 30  # the outlier dominates, which is the point


def test_itl_undefined_for_a_single_token():
    r = GenResult(model=REF, inter_token_ms=[])
    assert r.itl_p50_ms is None
    assert r.itl_stddev_ms is None


def test_think_ratio_reports_share_of_thinking_chunks():
    r = GenResult(
        model=REF,
        thinking_chunks=90,
        answer_chunks=10,
        server=ServerTiming(output_tokens=100, eval_ms=1000),
    )
    assert r.think_ratio == 0.9
