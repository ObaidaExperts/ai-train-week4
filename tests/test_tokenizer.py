from app.core.tokenizer import TokenCounter


def test_token_counter_chars() -> None:
    text = "Hello world"
    report = TokenCounter.get_token_report(text)
    assert report["characters"] == 11

def test_token_counter_gpt4() -> None:
    text = "Hello world"
    # "Hello" is 1 token, " world" is 1 token
    count = TokenCounter.count_openai_tokens(text, "gpt-4o")
    assert count == 2

def test_token_counter_gemini_estimate() -> None:
    text = "A" * 8
    # 8 / 4 = 2
    assert TokenCounter.estimate_gemini_tokens(text) == 2
