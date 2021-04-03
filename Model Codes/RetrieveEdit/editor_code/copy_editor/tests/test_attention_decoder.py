import torch

from editor_code.copy_editor.attention_decoder import AttentionDecoderCellOutput
from editor_code.copy_editor.vocab import HardCopyVocab, HardCopyDynamicVocab
from gtd.ml.torch.utils import assert_tensor_equal


class TestAttentionDecoderCellOutput(object):
    def test_transfer_copy_probs(self):

        # not actual probs, but that's ok
        probs = torch.FloatTensor([
            # special base    copy
            [1, 2, 3, 10, 11, 12, 14, 16],
            [4, 5, 6, 20, 21, 22, 24, 26],
        ])

        base_vocab = HardCopyVocab('a b'.split(), num_copy_tokens=3)
        dynamic_vocabs = [
            HardCopyDynamicVocab(base_vocab, 'b a c d e f g'.split()),  # d, e, f, g don't get assigned copy tokens (not enough)
            HardCopyDynamicVocab(base_vocab, 'e f f f a d'.split()),  # e, f, g get assigned copy tokens
        ]

        AttentionDecoderCellOutput._transfer_copy_probs(probs, dynamic_vocabs, base_vocab)

        assert_tensor_equal(probs, [
            [1, 2, 3, 24, 23, 0, 0, 16],  # copy prob for 'c' is not transferred, since it's not in base
            [4, 5, 6, 46, 21, 22, 24, 0],  # only prob for 'a' gets transferred
        ])
