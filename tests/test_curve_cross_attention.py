import math
import unittest

import torch
import torch.nn.functional as F

from models.dab_curve_detr import CurveCrossAttention, DABEncoder


def legacy_curve_cross_attention(module, query, key, value, key_padding_mask=None, attn_mask=None):
    batch_size, num_queries, _ = query.shape
    num_tokens = key.shape[1]
    q = query.view(batch_size, num_queries, module.nhead, module.head_dim * 2).transpose(1, 2)
    k = key.view(batch_size, num_tokens, module.nhead, module.head_dim * 2).transpose(1, 2)
    v = value.view(batch_size, num_tokens, module.nhead, module.head_dim).transpose(1, 2)

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(module.head_dim * 2))

    if attn_mask is not None:
        mask = attn_mask.to(device=scores.device)
        if mask.dtype == torch.bool:
            bias = torch.zeros(mask.shape, device=scores.device, dtype=scores.dtype)
            bias = bias.masked_fill(~mask, float('-inf'))
        else:
            bias = mask.to(dtype=scores.dtype)

        if bias.dim() == 2:
            bias = bias.unsqueeze(0).unsqueeze(0)
        elif bias.dim() == 3:
            bias = bias.unsqueeze(1)
        elif bias.dim() != 4:
            raise ValueError('Unsupported attn_mask shape for test')

        scores = scores + bias

    if key_padding_mask is not None:
        scores = scores.masked_fill(key_padding_mask[:, None, None, :], float('-inf'))

    attn = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn, v)
    output = output.transpose(1, 2).reshape(batch_size, num_queries, module.d_model)
    return F.linear(output, module.out_proj.weight, module.out_proj.bias)


class CurveCrossAttentionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.module = CurveCrossAttention(d_model=32, nhead=4, dropout=0.0)
        self.module.eval()
        self.batch_size = 2
        self.num_queries = 5
        self.num_tokens = 7
        self.query = torch.randn(self.batch_size, self.num_queries, 64)
        self.key = torch.randn(self.batch_size, self.num_tokens, 64)
        self.value = torch.randn(self.batch_size, self.num_tokens, 32)

    def assertCloseToLegacy(self, *, key_padding_mask=None, attn_mask=None):
        expected = legacy_curve_cross_attention(
            self.module,
            self.query,
            self.key,
            self.value,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
        )
        actual = self.module(
            self.query,
            self.key,
            self.value,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
        )
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    def test_matches_legacy_without_masks(self):
        self.assertCloseToLegacy()

    def test_matches_legacy_with_key_padding_mask(self):
        key_padding_mask = torch.tensor(
            [[False, False, False, True, True, False, False], [False, True, False, False, True, False, True]],
            dtype=torch.bool,
        )
        self.assertCloseToLegacy(key_padding_mask=key_padding_mask)

    def test_matches_legacy_with_bool_attn_mask(self):
        attn_mask = torch.ones(self.num_queries, self.num_tokens, dtype=torch.bool)
        attn_mask[:, -2:] = False
        self.assertCloseToLegacy(attn_mask=attn_mask)

    def test_matches_legacy_with_float_attn_mask_and_padding_mask(self):
        attn_mask = torch.zeros(self.batch_size, self.num_queries, self.num_tokens)
        attn_mask[0, 0, -1] = float('-inf')
        attn_mask[1, :, 1] = float('-inf')
        key_padding_mask = torch.tensor(
            [[False, False, False, False, False, True, False], [False, False, True, False, False, False, False]],
            dtype=torch.bool,
        )
        self.assertCloseToLegacy(key_padding_mask=key_padding_mask, attn_mask=attn_mask)


class GradientCheckpointingTest(unittest.TestCase):
    def _assert_module_matches_with_checkpointing(self, module_factory, inputs, output_fn):
        torch.manual_seed(0)
        base_module = module_factory(False)
        ckpt_module = module_factory(True)
        ckpt_module.load_state_dict(base_module.state_dict())
        base_module.train()
        ckpt_module.train()

        base_inputs = []
        ckpt_inputs = []
        for tensor in inputs:
            if tensor is None:
                base_inputs.append(None)
                ckpt_inputs.append(None)
            else:
                base_inputs.append(tensor.clone().detach().requires_grad_(tensor.requires_grad))
                ckpt_inputs.append(tensor.clone().detach().requires_grad_(tensor.requires_grad))

        base_outputs = output_fn(base_module, *base_inputs)
        ckpt_outputs = output_fn(ckpt_module, *ckpt_inputs)

        if isinstance(base_outputs, tuple):
            for actual, expected in zip(ckpt_outputs, base_outputs):
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
            base_loss = sum(output.sum() for output in base_outputs)
            ckpt_loss = sum(output.sum() for output in ckpt_outputs)
        else:
            torch.testing.assert_close(ckpt_outputs, base_outputs, atol=1e-6, rtol=1e-5)
            base_loss = base_outputs.sum()
            ckpt_loss = ckpt_outputs.sum()

        base_loss.backward()
        ckpt_loss.backward()

        for base_input, ckpt_input in zip(base_inputs, ckpt_inputs):
            if base_input is None or not base_input.requires_grad:
                continue
            torch.testing.assert_close(ckpt_input.grad, base_input.grad, atol=1e-6, rtol=1e-5)

        for (base_name, base_param), (ckpt_name, ckpt_param) in zip(base_module.named_parameters(), ckpt_module.named_parameters()):
            self.assertEqual(base_name, ckpt_name)
            if base_param.grad is None and ckpt_param.grad is None:
                continue
            torch.testing.assert_close(ckpt_param.grad, base_param.grad, atol=1e-6, rtol=1e-5)

    def test_encoder_gradient_checkpointing_matches_baseline(self):
        def module_factory(gradient_checkpointing):
            return DABEncoder(
                d_model=16,
                nhead=4,
                dim_feedforward=32,
                dropout=0.0,
                num_layers=2,
                gradient_checkpointing=gradient_checkpointing,
            )

        src = torch.randn(2, 6, 16, requires_grad=True)
        pos = torch.randn(2, 6, 16, requires_grad=True)
        key_padding_mask = torch.tensor(
            [[False, False, False, True, False, False], [False, True, False, False, False, True]],
            dtype=torch.bool,
        )

        self._assert_module_matches_with_checkpointing(
            module_factory,
            [src, pos, key_padding_mask],
            lambda module, src, pos, key_padding_mask: module(src, pos, key_padding_mask),
        )

if __name__ == '__main__':
    unittest.main()