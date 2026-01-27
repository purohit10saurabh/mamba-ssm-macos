import unittest

import torch
import torch.nn as nn

from mamba_ssm.modules.mamba2 import Mamba2


class TestMamba2(unittest.TestCase):
    def test_basic_forward(self):
        d_model, d_state, batch_size, seq_len = 64, 16, 2, 8
        model = Mamba2(d_model=d_model, d_state=d_state, d_conv=4, expand=2)
        x = torch.randn(batch_size, seq_len, d_model)
        out = model(x)
        self.assertEqual(out.shape, (batch_size, seq_len, d_model))

    def test_shape_consistency_large_model(self):
        d_model, d_state, headdim, batch_size, seq_len = 256, 64, 64, 2, 32
        model = Mamba2(d_model=d_model, d_state=d_state, headdim=headdim, d_conv=4, expand=2)
        x = torch.randn(batch_size, seq_len, d_model)
        out = model(x)
        self.assertEqual(out.shape, (batch_size, seq_len, d_model))
        self.assertEqual(model.d_inner, 512)
        self.assertEqual(model.nheads, 8)
        self.assertEqual(model.headdim, 64)

    def test_variable_length(self):
        d_model, d_state, batch_size = 64, 16, 2
        seq_lens = [5, 8]
        model = Mamba2(d_model=d_model, d_state=d_state, d_conv=4, expand=2)
        for seq_len in seq_lens:
            x = torch.randn(batch_size, seq_len, d_model)
            out = model(x)
            self.assertEqual(out.shape, (batch_size, seq_len, d_model))

    def test_different_headdim_configs(self):
        d_model, d_state, batch_size, seq_len = 192, 48, 2, 16
        for headdim in [32, 48, 64]:
            with self.subTest(headdim=headdim):
                model = Mamba2(d_model=d_model, d_state=d_state, headdim=headdim, d_conv=4, expand=2)
                expected_nheads = model.d_inner // headdim
                self.assertEqual(model.nheads, expected_nheads)
                x = torch.randn(batch_size, seq_len, d_model)
                out = model(x)
                self.assertEqual(out.shape, (batch_size, seq_len, d_model))

    def test_gradient_flow(self):
        d_model, d_state, batch_size, seq_len = 64, 16, 2, 8
        model = Mamba2(d_model=d_model, d_state=d_state, d_conv=4, expand=2)
        x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
        out = model(x)
        loss = nn.MSELoss()(out, torch.randn_like(out))
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(model.in_proj.weight.grad)
        self.assertIsNotNone(model.out_proj.weight.grad)
        self.assertIsNotNone(model.conv1d.weight.grad)
        self.assertIsNotNone(model.A_log.grad)
        self.assertIsNotNone(model.D.grad)
        self.assertIsNotNone(model.dt_bias.grad)

    def test_edge_case_dimensions(self):
        test_configs = [
            {"d_model": 32, "d_state": 8, "headdim": 16},
            {"d_model": 512, "d_state": 128, "headdim": 128},
            {"d_model": 96, "d_state": 24, "headdim": 24},
        ]
        for config in test_configs:
            with self.subTest(**config):
                model = Mamba2(d_model=config["d_model"], d_state=config["d_state"], headdim=config["headdim"], d_conv=4, expand=2)
                x = torch.randn(1, 4, config["d_model"])
                out = model(x)
                self.assertEqual(out.shape, (1, 4, config["d_model"]))


def run_mamba2_tests():
    print("🧪 Running Mamba2 tests...")
    try:
        suite = unittest.TestLoader().loadTestsFromTestCase(TestMamba2)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        if result.wasSuccessful():
            print("✅ All Mamba2 tests passed")
            return True
        print(f"❌ {len(result.failures)} failures, {len(result.errors)} errors")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False


if __name__ == "__main__":
    run_mamba2_tests()
