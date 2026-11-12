"""Temporary probe: proves branch protection refuses a merge while ci is red.

This file is never meant to reach main. It exists so the gate can be observed
failing on a real run rather than assumed from configuration.
"""
import unittest


class TestGateProof(unittest.TestCase):
    def test_this_fails_on_purpose(self):
        self.assertEqual("the gate blocks this", "deliberately different")
