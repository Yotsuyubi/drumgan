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

    def test_train_feature(self):
        y = np.random.rand(1, 1, 16384)
        sample = gan.train_feature(y, iteration=1)
        self.assertEqual(sample.shape, (1, 128))


if __name__ == '__main__':
    unittest.main()
