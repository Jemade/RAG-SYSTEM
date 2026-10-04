import pytest

from rag_system.evaluation import exact_match


@pytest.mark.parametrize(
    "answer,reference",
    [("0.85", "085"), ("-5", "5"), ("1.5", "15"), ("5%", "5"), ("+5", "5")],
)
def test_numeric_punctuation_changes_answer(answer, reference):
    assert exact_match(answer, [reference]) == 0


@pytest.mark.parametrize(
    "answer,reference",
    [(" 0.85. ", "0.85"), ("(-5)", "-5"), ("AES-256!", "aes256"), ("5%!", "5%")],
)
def test_sentence_punctuation_is_ignored(answer, reference):
    assert exact_match(answer, [reference]) == 1
