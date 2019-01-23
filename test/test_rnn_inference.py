#!/usr/bin/env python

import sys
import os
import unittest
import logging

import numpy as np

sys.path.append(os.path.abspath('../nematus'))
from nematus.rnn_inference import reshape_alignments


level = logging.DEBUG
logging.basicConfig(level=level, format='%(levelname)s: %(message)s')


class TestRNNInference(unittest.TestCase):

    def test_alignment_reshape(self):

        num_models = 1
        beam_size = 5
        batch_size = 2
        source_seq_len = 4
        target_seq_len = 3

        a = np.arange(1 * 5 * 2 * 3 * 4).reshape(target_seq_len, num_models, beam_size * batch_size, source_seq_len)

        expected_shape = (2, 1, 5, 3, 4)
        expected_array = np.array([[[[[  0,   1,   2,   3],
          [ 40,  41,  42,  43],
          [ 80,  81,  82,  83]],

         [[  8,   9,  10,  11],
          [ 48,  49,  50,  51],
          [ 88,  89,  90,  91]],

         [[ 16,  17,  18,  19],
          [ 56,  57,  58,  59],
          [ 96,  97,  98,  99]],

         [[ 24,  25,  26,  27],
          [ 64,  65,  66,  67],
          [104, 105, 106, 107]],

         [[ 32,  33,  34,  35],
          [ 72,  73,  74,  75],
          [112, 113, 114, 115]]]],


       [[[[  4,   5,   6,   7],
          [ 44,  45,  46,  47],
          [ 84,  85,  86,  87]],

         [[ 12,  13,  14,  15],
          [ 52,  53,  54,  55],
          [ 92,  93,  94,  95]],

         [[ 20,  21,  22,  23],
          [ 60,  61,  62,  63],
          [100, 101, 102, 103]],

         [[ 28,  29,  30,  31],
          [ 68,  69,  70,  71],
          [108, 109, 110, 111]],

         [[ 36,  37,  38,  39],
          [ 76,  77,  78,  79],
          [116, 117, 118, 119]]]]])

        actual_array = reshape_alignments(a, beam_size)

        self.assertTrue(np.allclose(expected_array, actual_array))

        self.assertEqual(expected_shape, actual_array.shape)



if __name__ == '__main__':
    unittest.main()
