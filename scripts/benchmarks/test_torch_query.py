"""CPU-only checks for the capture wrappers of both encoder types and output validation."""

import unittest

import numpy as np
import torch
from transformers import BertConfig, BertModel, ModernBertConfig, ModernBertModel

from bench_torch_query import (bert_embeddings, masked_mean_normalize, modernbert_forward, summarize_attention_trace,
                               validate_outputs)


class TorchQueryTests(unittest.TestCase):
    def test_specialized_mask_matches_ordinary_forward(self):
        self.assertFalse(torch.cuda.is_initialized())
        torch.manual_seed(1)
        config = BertConfig(vocab_size=32, hidden_size=16, num_hidden_layers=1,
                            num_attention_heads=2, intermediate_size=32,
                            hidden_dropout_prob=0., attention_probs_dropout_prob=0.)
        config._attn_implementation = "sdpa"
        model = BertModel(config, add_pooling_layer=False).eval()
        for mask in ([[1, 1, 1]], [[1, 1, 0]]):
            inputs = {"input_ids": torch.tensor([[1, 2, 3]]),
                      "token_type_ids": torch.zeros((1, 3), dtype=torch.long),
                      "attention_mask": torch.tensor(mask)}
            with torch.inference_mode():
                actual = bert_embeddings(model, inputs, all(x == 1 for x in mask[0]))
                wanted = masked_mean_normalize(model(**inputs, return_dict=False)[0], inputs["attention_mask"])
            torch.testing.assert_close(actual, wanted)
        self.assertFalse(torch.cuda.is_initialized())

    def test_modernbert_with_its_masks_made_beforehand_matches_the_ordinary_forward(self):
        torch.manual_seed(1)
        # Three layers: the first sees the whole text, the others a window of four tokens.
        config = ModernBertConfig(vocab_size=32, hidden_size=16, num_hidden_layers=3, num_attention_heads=2,
                                  intermediate_size=24, global_attn_every_n_layers=3, local_attention=4,
                                  max_position_embeddings=32, pad_token_id=0, reference_compile=False)
        config._attn_implementation = "sdpa"
        model = ModernBertModel(config).eval()
        ids = torch.arange(1, 13).unsqueeze(0)
        for mask in (torch.ones_like(ids), torch.tensor([[1] * 9 + [0] * 3])):
            inputs = {"input_ids": ids, "attention_mask": mask}
            forward = modernbert_forward(model, inputs, bool(mask.all()))
            other = {"input_ids": ids.flip(1), "attention_mask": mask}
            with torch.inference_mode():
                for fresh in (inputs, other):
                    wanted = masked_mean_normalize(model(**fresh, return_dict=False)[0], mask)
                    torch.testing.assert_close(forward(fresh), wanted)

    def test_masked_pooling_and_output_checks(self):
        result = masked_mean_normalize(torch.tensor([[[3., 4.], [999., 999.]]]),
                                       torch.tensor([[1, 0]])).numpy()
        np.testing.assert_allclose(result, [[.6, .8]], rtol=1e-6)
        validate_outputs(result, 1, 2)
        with self.assertRaises(RuntimeError):
            validate_outputs(np.array([[0., 0.]], np.float32), 1, 2)

    def test_profile_needs_sdpa_and_gpu_kernel_evidence(self):
        with self.assertRaisesRegex(RuntimeError, "both SDPA"):
            summarize_attention_trace({"traceEvents": [{"cat": "cpu_op", "name": "aten::scaled_dot_product_attention"}]})


if __name__ == "__main__":
    unittest.main(verbosity=2)
