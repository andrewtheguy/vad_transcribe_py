import pytest

import whisper_transcribe_py.audio_transcriber as audio_transcriber


class TimeStub:
    def __init__(self, start: float = 0.0) -> None:
        self.value = start

    def time(self) -> float:
        return self.value

    def advance(self, delta: float) -> None:
        self.value += delta


@pytest.fixture
def time_stub(monkeypatch):
    stub = TimeStub(start=100.0)
    monkeypatch.setattr(audio_transcriber.time, "time", stub.time)
    return stub


def test_try_add_under_limit_sets_initial_timestamp(time_stub):
    limiter = audio_transcriber.QueueBacklogLimiter(max_seconds=10.0)

    assert limiter.initial_timestamp is None
    accepted = limiter.try_add(5.0)

    assert accepted is True
    assert limiter.initial_timestamp == pytest.approx(time_stub.value)
    assert limiter.current_seconds == pytest.approx(5.0)


def test_try_add_exceeding_limit_triggers_drop_callback_once(time_stub):
    limiter = audio_transcriber.QueueBacklogLimiter(max_seconds=5.0)
    drop_events: list[float] = []
    limiter.register_drop_callback(lambda ts: drop_events.append(ts))

    first_drop = limiter.try_add(6.0)

    assert first_drop is False
    assert drop_events == pytest.approx([100.0])
    assert limiter.dropped_seconds == pytest.approx(6.0)
    assert limiter._drop_notice_active is True  # type: ignore[attr-defined]

    time_stub.advance(1.0)
    second_attempt = limiter.try_add(1.0)

    assert second_attempt is True
    assert drop_events == pytest.approx([100.0])
    assert limiter._drop_notice_active is False  # type: ignore[attr-defined]


def test_drop_mode_clears_after_consuming_below_resume(time_stub):
    limiter = audio_transcriber.QueueBacklogLimiter(max_seconds=10.0, resume_seconds=4.0)
    drop_events: list[float] = []
    limiter.register_drop_callback(lambda ts: drop_events.append(ts))

    assert limiter.try_add(6.0) is True
    time_stub.advance(1.0)
    assert limiter.try_add(6.0) is False  # enters drop mode while backlog>resume
    assert drop_events == pytest.approx([101.0])

    time_stub.advance(1.0)
    assert limiter.try_add(1.0) is False  # still dropping because backlog>resume
    assert drop_events == pytest.approx([101.0])

    limiter.consume(3.5)  # backlog now below resume threshold
    assert limiter.current_seconds == pytest.approx(2.5)

    assert limiter.try_add(2.0) is True
    assert drop_events == pytest.approx([101.0])


def test_drop_callback_uses_current_time(time_stub):
    limiter = audio_transcriber.QueueBacklogLimiter(max_seconds=5.0, initial_timestamp=90.0)
    drop_events: list[float] = []
    limiter.register_drop_callback(lambda ts: drop_events.append(ts))

    assert limiter.try_add(4.0) is True
    time_stub.advance(2.0)
    assert limiter.try_add(4.0) is False

    assert drop_events == pytest.approx([time_stub.value])


# Test removed: pending_chunk_start_timestamp method was removed as part of timestamp refactoring
# Timestamp estimation is no longer needed since all input sources provide real timestamps
# def test_pending_chunk_start_timestamp_accounts_for_progress_and_drops(time_stub):
#     limiter = audio_transcriber.QueueBacklogLimiter(max_seconds=10.0, initial_timestamp=100.0)
#
#     assert limiter.try_add(4.0) is True
#     limiter.note_timestamp_progress(1.5)
#
#     time_stub.advance(1.0)
#     assert limiter.try_add(8.0) is False  # dropped chunk tracked for timestamp math
#
#     time_stub.value = 110.0
#     estimate = limiter.pending_chunk_start_timestamp()
#
#     assert estimate == pytest.approx(109.5)


def test_note_timestamp_progress_caps_at_total_accounted(time_stub):
    limiter = audio_transcriber.QueueBacklogLimiter(max_seconds=None)
    assert limiter.try_add(3.0) is True

    limiter.note_timestamp_progress(10.0)
    limiter.note_timestamp_progress(5.0)

    assert limiter._timestamp_consumed_seconds == pytest.approx(3.0)  # type: ignore[attr-defined]
