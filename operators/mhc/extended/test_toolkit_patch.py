import unittest
from patch_deepjit_host import patch_text, OLD, NEW

class ToolkitPatchTests(unittest.TestCase):
    def test_exact_patch_and_idempotence(self):
        source = '#include <vector>\n' + OLD + '\nreturn toolkit;'
        expected = '#include <vector>\n' + NEW + '\nreturn toolkit;'
        self.assertEqual(patch_text(source), expected)
        self.assertEqual(patch_text(expected), expected)
    def test_unknown_source_is_rejected(self):
        with self.assertRaises(RuntimeError):
            patch_text('different upstream implementation')
    def test_ambiguous_context_is_rejected(self):
        with self.assertRaises(RuntimeError):
            patch_text(OLD + '\n' + OLD)

if __name__ == '__main__':
    unittest.main()
