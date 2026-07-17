from __future__ import annotations

import pytest

from quant_trader.core.models import ReviewAction
from quant_trader.llm.parsing import LLMResponseError, parse_review

VALID = (
    '{"action":"reduce","weight_multiplier":0.5,"confidence":0.6,'
    '"thesis":"trend","risks":["volatility"],"invalidation":"SMA break",'
    '"input_anomalies":[]}'
)


@pytest.mark.parametrize(
    "content",
    [VALID, f"  ```json\n{VALID}\n```  ", f"<think>{VALID}</think>\n```json\n{VALID}\n```"],
)
def test_parse_review_accepts_one_model_json_object(content: str) -> None:
    review = parse_review(content)

    assert review.action is ReviewAction.REDUCE
    assert review.weight_multiplier == 0.5
    assert review.risks == ("volatility",)


@pytest.mark.parametrize(
    "content",
    [
        "",
        "<think>reasoning",
        f"<think>{VALID}</think>{VALID}{VALID}",
        f"{VALID}\n{VALID}",
        f"{VALID}\ncommentary",
        f"```\n{VALID}\n```",
        f"```json\n{VALID}\n```\nmore",
        "[]",
        "true",
        VALID.replace('"input_anomalies"', '"anomalies"'),
        VALID.replace('"weight_multiplier":0.5', '"weight_multiplier":1.1'),
        VALID.replace('"confidence":0.6', '"confidence":true'),
        VALID.replace('"thesis":"trend"', '"thesis":""'),
        VALID.replace('"trend"', '"bad\x01text"'),
        VALID.replace('"trend"', r'"bad\u0001text"'),
    ],
)
def test_parse_review_rejects_unsafe_or_invalid_responses(content: str) -> None:
    with pytest.raises(LLMResponseError) as error:
        parse_review(content)

    assert VALID not in str(error.value)
