from app.services.transcriber import normalize_transcript


def test_normalizes_whisper_traditional_chinese_output() -> None:
    transcript = "參加宴請，關鍵人物是比亞迪股份有限公司董事長王傳福，關注儲能業務。"

    assert normalize_transcript(transcript) == (
        "参加宴请，关键人物是比亚迪股份有限公司董事长王传福，关注储能业务。"
    )
