import pytest

from langchain.tools import ToolException

from langchain_project.tools.calculator import safe_eval_arithmetic


def test_basic_arithmetic():
    assert safe_eval_arithmetic("2 + 3 * 4") == 14


def test_rejects_unsupported_syntax():
    with pytest.raises(ToolException):
        safe_eval_arithmetic("__import__('os').system('ls')")


def test_rejects_huge_exponent():
    with pytest.raises(ToolException):
        safe_eval_arithmetic("9 ** 9 ** 9")
