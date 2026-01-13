from __future__ import annotations

from tools.webapp.scripts.import_time2score import (
    _compute_end_from_start_and_delta,
    _infer_end_period_for_within_times,
)


def should_infer_end_period_when_tts_end_time_wraps_remaining() -> None:
    # P1 1:00 -> end 14:00 means the penalty ends in P2 (clock counts down).
    per_end = _infer_end_period_for_within_times(
        start_period=1,
        start_within_s=60,
        end_within_s=14 * 60,
        mode="remaining",
        period_len_s=15 * 60,
    )
    assert per_end == 2


def should_not_advance_end_period_for_normal_remaining_penalty() -> None:
    per_end = _infer_end_period_for_within_times(
        start_period=1,
        start_within_s=12 * 60,
        end_within_s=10 * 60,
        mode="remaining",
        period_len_s=15 * 60,
    )
    assert per_end == 1


def should_compute_end_from_start_and_delta_with_wrap_remaining() -> None:
    # P1 1:00 remaining + 2:00 penalty -> P2 14:00 remaining.
    per_end, end_s = _compute_end_from_start_and_delta(
        start_period=1,
        start_within_s=60,
        delta_s=120,
        mode="remaining",
        period_len_s=15 * 60,
    )
    assert per_end == 2
    assert end_s == 14 * 60


def should_compute_end_from_start_and_delta_with_wrap_elapsed() -> None:
    # P1 14:00 elapsed + 2:00 penalty -> P2 1:00 elapsed.
    per_end, end_s = _compute_end_from_start_and_delta(
        start_period=1,
        start_within_s=14 * 60,
        delta_s=120,
        mode="elapsed",
        period_len_s=15 * 60,
    )
    assert per_end == 2
    assert end_s == 60
