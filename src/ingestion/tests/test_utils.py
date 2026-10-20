"""Unit tests cho module utils — tiện ích dùng chung."""
import pytest
from src.ingestion.utils import clean_whitespace


class TestCleanWhitespace:
    """Kiểm tra hàm chuẩn hóa khoảng trắng."""

    def test_basic_cleaning(self):
        text = "  Hello  \n  World  "
        assert clean_whitespace(text) == "Hello\nWorld"

    def test_multiple_blank_lines_collapsed(self):
        text = "Line 1\n\n\n\n\nLine 2"
        result = clean_whitespace(text)
        assert result == "Line 1\n\nLine 2"

    def test_leading_trailing_blanks_removed(self):
        text = "\n\n\nContent\n\n\n"
        assert clean_whitespace(text) == "Content"

    def test_single_blank_line_preserved(self):
        text = "Para 1\n\nPara 2"
        assert clean_whitespace(text) == "Para 1\n\nPara 2"

    def test_empty_string(self):
        assert clean_whitespace("") == ""

    def test_only_whitespace(self):
        assert clean_whitespace("   \n  \n   ") == ""

    def test_tabs_and_spaces_stripped(self):
        text = "\tHello\t\n  World  "
        assert clean_whitespace(text) == "Hello\nWorld"
