from pipeline.audio_analyzer import _normalize_whisper_result


def test_normalize_whisper_result_keeps_word_level_timestamps():
    result = {
        "text": "The Vikings sailed west.",
        "language": "en",
        "segments": [
            {
                "start": 0.0,
                "end": 2.4,
                "text": " The Vikings sailed west.",
                "words": [
                    {"word": " The", "start": 0.0, "end": 0.3},
                    {"word": " Vikings", "start": 0.31, "end": 0.9},
                    {"word": " sailed", "start": 1.0, "end": 1.5},
                    {"word": " west.", "start": 1.6, "end": 2.2},
                ],
            }
        ],
    }

    transcript, words, segments = _normalize_whisper_result(result, 2.5)

    assert transcript == "The Vikings sailed west."
    assert [word["text"] for word in words] == ["The", "Vikings", "sailed", "west."]
    assert words[0]["start"] == 0.0
    assert words[-1]["end"] == 2.2
    assert segments == [{"start": 0.0, "end": 2.4, "text": "The Vikings sailed west."}]


def test_normalize_whisper_result_clamps_word_end_to_audio_duration():
    result = {
        "text": "Hello",
        "segments": [
            {
                "start": 0.0,
                "end": 4.0,
                "text": " Hello",
                "words": [{"word": " Hello", "start": 3.5, "end": 4.5}],
            }
        ],
    }

    transcript, words, segments = _normalize_whisper_result(result, 4.0)

    assert transcript == "Hello"
    assert words == [{"text": "Hello", "start": 3.5, "end": 4.0}]
    assert segments == [{"start": 0.0, "end": 4.0, "text": "Hello"}]
