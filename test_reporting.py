
import unittest, tempfile
from pathlib import Path
import kit
class ReportingTests(unittest.TestCase):
    def test_zero_positive_metrics(self):
        result=kit.classification([0,0],[0,0])
        self.assertEqual(result['accuracy'],1)
        self.assertEqual(result['f1'],0)
    def test_html_escapes_untrusted_content(self):
        original=kit.ROOT
        try:
            with tempfile.TemporaryDirectory(dir=original) as folder:
                kit.ROOT=Path(folder)
                kit.report('<script>bad()</script>',{'source':'<img src=x onerror=bad()>'})
                output=(Path(folder)/'results/report.html').read_text()
                self.assertNotIn('<script>bad()',output)
                self.assertIn('&lt;img',output)
        finally: kit.ROOT=original
