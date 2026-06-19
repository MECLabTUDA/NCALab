from ncalab import string_ellipsis, Color


def test_string_ellipsis():
    assert string_ellipsis("hello world", max_len=100) == "hello world"
    assert string_ellipsis("hello world", max_len=(len("hello "))) == "HW"
    assert string_ellipsis("helloworld", max_len=3) == "he…"


def test_color():
    rgba4f = Color((0.1, 0.3, 0.3, 0.7))
    assert rgba4f.rgba4b == (int(0.1 * 255), int(0.3 * 255), int(0.3 * 255), int(0.7 * 255))
