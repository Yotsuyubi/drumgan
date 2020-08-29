# -*- coding: utf-8 -*-
import unittest
import numpy as np
from .context import drumgan
gan = drumgan.DrumGAN()


class BasicTestSuite(unittest.TestCase):

    def test_generate(self):
        z = np.random.rand(1, 128)
        sample, z_out = gan.generate(z)
        self.assertEqual(sample.shape, (16384,))
        self.assertEqual(z_out.shape, (1, 128))

    def test_random_generate(self):
        sample, z_out = gan.random_generate()
        self.assertEqual(sample.shape, (16384,))
        self.assertEqual(z_out.shape, (1, 128))

    def test_optim_feature(self):
        y = np.random.rand(1, 1, 16384)
        z = gan.optim_feature(y, iteration=1)
        self.assertEqual(z.shape, (1, 128))


if __name__ == '__main__':
    unittest.main()
