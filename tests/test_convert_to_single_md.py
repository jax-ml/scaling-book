#!/usr/bin/env python3
"""Regression tests for bin/convert_to_single_md.py.

Run with: python3 tests/test_convert_to_single_md.py
"""

import importlib.util
import unittest
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "bin" / "convert_to_single_md.py"
_spec = importlib.util.spec_from_file_location("convert_to_single_md", SCRIPT_PATH)
convert_to_single_md = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(convert_to_single_md)


class ConvertFigureIncludesTest(unittest.TestCase):
    def test_caption_with_escaped_quotes_is_not_truncated(self):
        # Mirrors the real figure.liquid tag in training.md for
        # assets/img/deepseek-pipeline.png, which embeds an <a href=...>
        # using backslash-escaped quotes inside the caption attribute.
        block = (
            '{% include figure.liquid path="assets/img/x.png" class="img-fluid" '
            'caption="<b>Figure:</b> see <a href=\\"https://example.com\\">paper</a>. '
            'We avoid \\"stranding\\" FLOPs." %}'
        )
        result = convert_to_single_md.convert_figure_includes(block)
        self.assertEqual(
            result,
            '![Figure: see paper. We avoid "stranding" FLOPs.](assets/img/x.png)',
        )

    def test_caption_without_escaped_quotes_still_works(self):
        block = (
            '{% include figure.liquid path="assets/img/y.png" class="img-small" '
            'caption="A plain caption." %}'
        )
        result = convert_to_single_md.convert_figure_includes(block)
        self.assertEqual(result, "![A plain caption.](assets/img/y.png)")


if __name__ == "__main__":
    unittest.main()
