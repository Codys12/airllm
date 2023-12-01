import sys
import unittest

import torch
sys.path.insert(0, '../airllm')

from airllm import compress_layer_state_dict, uncompress_layer_state_dict




class TestCompression(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass

    def test_should_compress_uncompress(self):
        torch.manual_seed(0)
        a0 = torch.normal(0, 1, (32, 128), dtype=torch.float16).cuda()
        a1 = torch.normal(0, 1, (32, 128), dtype=torch.float16).cuda()

        a_state_dict = {'a0':a0, 'a1':a1}

        loss_fn = torch.nn.MSELoss()

        for compression in [None, '4bit', '8bit']:
            b = compress_layer_state_dict(a_state_dict, compression)

            print(f"for compression {compression}, compressed to: { {k:v.shape for k,v in b.items()} }")

            aa = uncompress_layer_state_dict(b)

            for k in aa.keys():

                if compression is None:
                    self.assertAlmostEqual(aa[k], a_state_dict[k])
                else:
                    RMSE_loss = torch.sqrt(loss_fn(aa[k], a_state_dict[k]))
                    self.assertLess(RMSE_loss.detach().numpy()[0], 0.5)