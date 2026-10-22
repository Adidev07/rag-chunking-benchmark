import unittest
from app import *
class DomainTests(unittest.TestCase):
    def test_complete_coverage(self):
        words=[str(i) for i in range(35)];rows=chunks([{'id':'d','text':' '.join(words)}],12,3);self.assertEqual(set(' '.join(r['text'] for r in rows).split()),set(words))
    def test_bad_overlap(self):
        with self.assertRaises(ValueError):chunks([],8,8)
    def test_known_source(self):
        r=benchmark([{'id':'a','text':'refund policy'}],[{'query':'refund','source':'a'}],12,0);self.assertEqual(r['mrr'],1)
