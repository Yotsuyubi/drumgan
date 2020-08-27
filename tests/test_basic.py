# -*- coding: utf-8 -*-

from .context import drumgan
gan = drumgan.DrumGAN()

import unittest
import numpy as np


class BasicTestSuite(unittest.TestCase):

    def test_generate(self):
        z = np.random.rand(1, 128)
        sample = gan.generate(z)
        self.assertEqual(sample.shape, (16384,))

    def test_random_generate(self):
        sample = gan.random_generate()
        self.assertEqual(sample.shape, (16384,))


if __name__ == '__main__':
    unittest.main()
