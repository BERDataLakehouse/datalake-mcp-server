"""Tests for the argument checker module."""

import pytest

from src.service.arg_checkers import contains_control_characters, not_falsy


class TestNotFalsy:
    """Tests for not_falsy function."""

    def test_truthy_string(self):
        """Test that truthy string passes."""
        result = not_falsy("hello", "test_arg")
        assert result == "hello"

    def test_truthy_number(self):
        """Test that truthy number passes."""
        result = not_falsy(42, "test_arg")
        assert result == 42

    def test_truthy_list(self):
        """Test that non-empty list passes."""
        result = not_falsy([1, 2, 3], "test_arg")
        assert result == [1, 2, 3]

    def test_truthy_dict(self):
        """Test that non-empty dict passes."""
        result = not_falsy({"key": "value"}, "test_arg")
        assert result == {"key": "value"}

    def test_none_raises(self):
        """Test that None raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            not_falsy(None, "my_argument")

        assert "my_argument is required" in str(exc_info.value)

    def test_empty_string_raises(self):
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            not_falsy("", "my_argument")

        assert "my_argument is required" in str(exc_info.value)

    def test_zero_raises(self):
        """Test that zero raises ValueError (falsy)."""
        with pytest.raises(ValueError) as exc_info:
            not_falsy(0, "my_argument")

        assert "my_argument is required" in str(exc_info.value)

    def test_empty_list_raises(self):
        """Test that empty list raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            not_falsy([], "my_argument")

        assert "my_argument is required" in str(exc_info.value)

    def test_empty_dict_raises(self):
        """Test that empty dict raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            not_falsy({}, "my_argument")

        assert "my_argument is required" in str(exc_info.value)

    def test_false_raises(self):
        """Test that False raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            not_falsy(False, "my_argument")

        assert "my_argument is required" in str(exc_info.value)


class TestContainsControlCharacters:
    """Tests for contains_control_characters function."""

    def test_normal_string_returns_negative_one(self):
        """Test that normal string without control chars returns -1."""
        result = contains_control_characters("Hello, World!")
        assert result == -1

    def test_string_with_newline_returns_position(self):
        """Test that string with newline returns its position."""
        result = contains_control_characters("Hello\nWorld")
        assert result == 5  # Position of \n

    def test_string_with_tab_returns_position(self):
        """Test that string with tab returns its position."""
        result = contains_control_characters("Hello\tWorld")
        assert result == 5  # Position of \t

    def test_string_with_carriage_return(self):
        """Test that string with carriage return returns its position."""
        result = contains_control_characters("Hello\rWorld")
        assert result == 5  # Position of \r

    def test_string_with_null_char(self):
        """Test that string with null character returns its position."""
        result = contains_control_characters("Hello\x00World")
        assert result == 5  # Position of null char

    def test_control_char_at_start(self):
        """Test control character at the start of string."""
        result = contains_control_characters("\x00Hello")
        assert result == 0

    def test_control_char_at_end(self):
        """Test control character at the end of string."""
        result = contains_control_characters("Hello\x00")
        assert result == 5

    def test_allowed_chars_ignored(self):
        """Test that allowed control characters are ignored."""
        result = contains_control_characters("Hello\nWorld", allowed_chars=["\n"])
        assert result == -1

    def test_multiple_allowed_chars(self):
        """Test multiple allowed control characters."""
        result = contains_control_characters(
            "Hello\n\tWorld", allowed_chars=["\n", "\t"]
        )
        assert result == -1

    def test_some_allowed_some_not(self):
        """Test mix of allowed and disallowed control characters."""
        # \n is allowed but \t is not
        result = contains_control_characters("Hello\n\tWorld", allowed_chars=["\n"])
        assert result == 6  # Position of \t

    def test_empty_string_returns_negative_one(self):
        """Test that empty string returns -1."""
        result = contains_control_characters("")
        assert result == -1

    def test_unicode_string_without_control(self):
        """Test unicode string without control characters."""
        result = contains_control_characters("Héllo Wörld 你好")
        assert result == -1

    def test_bell_character(self):
        """Test string with bell character (\\a)."""
        result = contains_control_characters("Hello\aWorld")
        assert result == 5  # Position of \a (bell)

    def test_backspace_character(self):
        """Test string with backspace character."""
        result = contains_control_characters("Hello\bWorld")
        assert result == 5  # Position of \b

    def test_form_feed_character(self):
        """Test string with form feed character."""
        result = contains_control_characters("Hello\fWorld")
        assert result == 5  # Position of \f

    def test_vertical_tab_character(self):
        """Test string with vertical tab character."""
        result = contains_control_characters("Hello\vWorld")
        assert result == 5  # Position of \v

    def test_escape_character(self):
        """Test string with escape character."""
        result = contains_control_characters("Hello\x1bWorld")  # ESC character
        assert result == 5

    def test_none_allowed_chars_treated_as_empty(self):
        """Test that None allowed_chars is treated as empty list."""
        result = contains_control_characters("Hello\nWorld", allowed_chars=None)
        assert result == 5  # \n should be detected
