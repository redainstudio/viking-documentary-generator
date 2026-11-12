from pipeline.scene_planner import build_timed_windows


def test_windows_preserve_words_and_respect_max_duration():
    words = []
    cursor = 0.0
    for i in range(80):
        words.append({"text": f"word{i}", "start": cursor, "end": cursor + 0.12})
        cursor += 0.18
    windows = build_timed_windows(words, max_seconds=4.8, min_seconds=2.0)
    assert windows
    assert all(item["duration"] <= 4.8 for item in windows)
    assert " ".join(item["narration"] for item in windows).split() == [f"word{i}" for i in range(80)]


def test_sentence_boundary_emits_window():
    words = [
        {"text": "Hello.", "start": 0.0, "end": 0.3},
        {"text": "Next", "start": 0.5, "end": 0.8},
        {"text": "sentence.", "start": 0.9, "end": 1.4},
    ]
    result = build_timed_windows(words, max_seconds=4.8, min_seconds=0.2)
    assert len(result) == 2
    assert result[0]["narration"] == "Hello."


def test_hard_cap_five_seconds():
    words = [
        {"text": "A", "start": 0.0, "end": 0.4},
        {"text": "long", "start": 4.9, "end": 5.2},
    ]
    result = build_timed_windows(words, max_seconds=5.0, min_seconds=1.0)
    assert all(scene["duration"] <= 5.0 for scene in result)


def test_windows_cover_entire_audio_timeline_without_gaps():
    words = [
        {"text": "First.", "start": 0.2, "end": 0.7},
        {"text": "Second", "start": 1.0, "end": 1.4},
        {"text": "part.", "start": 1.5, "end": 1.9},
        {"text": "End", "start": 9.5, "end": 9.9},
    ]
    result = build_timed_windows(words, max_seconds=4.8, min_seconds=1.0, audio_duration=10.0)
    assert result[0]["start"] == 0.0
    assert result[-1]["end"] == 10.0
    for left, right in zip(result, result[1:]):
        assert abs(left["end"] - right["start"]) < 0.002
    assert all(0 < item["duration"] <= 4.8 for item in result)
