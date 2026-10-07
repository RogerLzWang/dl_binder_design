import math
import unittest

import numpy as np

import af2_scores


class AF2ScoresTest(unittest.TestCase):

    def setUp(self):
        self.pae = np.full((4, 4), 30.0)
        self.pae[:2, 2:] = [[2.0, 4.0], [12.0, 3.0]]
        self.pae[2:, :2] = [[5.0, 15.0], [2.0, 2.0]]
        self.plddt = np.array([90.0, 80.0, 70.0, 60.0])
        self.positions = np.zeros((4, 37, 3))
        self.positions[:, 3, 0] = [0.0, 1.0, 3.0, 4.0]
        self.mask = np.ones((4, 37))

    def test_ipsae_takes_maximum_direction(self):
        # With d0=1, the last reverse-direction row contains two PAE=2 pairs,
        # each of which contributes 1 / (1 + 2**2) = 0.2.
        self.assertAlmostEqual(
            af2_scores.calculate_ipsae(self.pae, binder_length=2), 0.2
        )

    def test_pdockq_matches_published_formula(self):
        contacts = np.ones((2, 2), dtype=bool)
        interface = np.ones(4, dtype=bool)
        x = np.mean(self.plddt) * math.log10(4)
        expected = 0.724 / (1 + math.exp(-0.052 * (x - 152.611))) + 0.018
        self.assertAlmostEqual(
            af2_scores.calculate_pdockq(self.plddt, contacts, interface),
            expected,
        )

    def test_pdockq2_matches_better_direction(self):
        contacts = np.ones((2, 2), dtype=bool)
        interface = np.ones(4, dtype=bool)
        mean_plddt = np.mean(self.plddt)
        directional_scores = []
        for block in (self.pae[:2, 2:], self.pae[2:, :2]):
            mean_ptm = np.mean(1.0 / (1.0 + np.square(block / 10.0)))
            x = mean_plddt * mean_ptm
            directional_scores.append(
                1.31 / (1 + math.exp(-0.075 * (x - 84.733))) + 0.005
            )
        self.assertAlmostEqual(
            af2_scores.calculate_pdockq2(
                self.pae, self.plddt, 2, contacts, interface
            ),
            max(directional_scores),
        )

    def test_scores_are_zero_without_contacts(self):
        self.positions[2:, 3, 0] += 100.0
        result = af2_scores.calculate_scores(
            self.pae, self.plddt, self.positions, self.mask, binder_length=2
        )
        self.assertEqual(result["pdockq"], 0.0)
        self.assertEqual(result["pdockq2"], 0.0)
        self.assertGreater(result["ipsae"], 0.0)

    def test_missing_cb_falls_back_to_ca(self):
        self.mask[:, 3] = 0.0
        self.positions[:, 1, 0] = [0.0, 1.0, 3.0, 4.0]
        result = af2_scores.calculate_scores(
            self.pae, self.plddt, self.positions, self.mask, binder_length=2
        )
        self.assertGreater(result["pdockq"], 0.0)
        self.assertGreater(result["pdockq2"], 0.0)

    def test_invalid_chain_split_is_rejected(self):
        with self.assertRaises(ValueError):
            af2_scores.calculate_ipsae(self.pae, binder_length=4)


if __name__ == "__main__":
    unittest.main()
