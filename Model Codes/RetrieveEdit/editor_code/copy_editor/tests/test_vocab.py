import pytest
import numpy as np

from editor_code.copy_editor.vocab import HardCopyVocab, DynamicMultiVocabTokenEmbedder, HardCopyDynamicVocab, base_plus_copy_indices
from gtd.ml.torch.token_embedder import TokenEmbedder
from gtd.ml.torch.utils import assert_tensor_equal
from gtd.ml.vocab import SimpleEmbeddings, WordVocab


class TestDynamicMultiVocabTokenEmbedder(object):
    @pytest.fixture
    def base_vocab(self):
        num_copy_tokens = 5
        tokens = 'a b c d e'.split()
        return HardCopyVocab(tokens, num_copy_tokens)

    @pytest.fixture
    def dynamic_vocabs(self, base_vocab):
        return [
            HardCopyDynamicVocab(base_vocab, 'a b c d e'.split()),  # a, b, c, d, e
            HardCopyDynamicVocab(base_vocab, 'c c e d z'.split()),  # c, e, d, z
        ]

    @pytest.fixture
    def embeds_array(self, base_vocab):
        embed_dim = 2
        vocab_size = len(base_vocab)

        # 0  1   0: <unk>
        # 2  3   1: <start>
        # 4  5   2: <stop>
        # 6  7   3: a
        # 8  9   4: b
        # 10 11  5: c
        # 12 13  6: d
        # 14 15  7: e
        # 16 17  8: <copy0>
        # 18 19  9: <copy1>
        # 20 21  10: <copy2>
        # 22 23  11: <copy3>
        # 24 25  12: <copy4>

        return np.reshape(np.arange(vocab_size * embed_dim), (vocab_size, embed_dim)).astype(np.float32)

    @pytest.fixture
    def token_embedder(self, base_vocab, embeds_array, dynamic_vocabs):
        word_embeds = SimpleEmbeddings(embeds_array, base_vocab)
        base_embedder = TokenEmbedder(word_embeds)
        return DynamicMultiVocabTokenEmbedder(base_embedder, dynamic_vocabs, base_vocab)

    @pytest.fixture
    def target_words(self):
        return [
            'c b c c'.split(),
            'z b c'.split(),
        ]

    @pytest.fixture
    def multi_vocab_indices(self, target_words, dynamic_vocabs, base_vocab):
        return base_plus_copy_indices(target_words, dynamic_vocabs, base_vocab)

    def test_base_plus_copy_indices(self, multi_vocab_indices):
        assert_tensor_equal(multi_vocab_indices.mask, [
            [1, 1, 1, 1],
            [1, 1, 1, 0],
        ])

        assert_tensor_equal(multi_vocab_indices.values, [
            [[5, 10], [4, 9], [5, 10], [5, 10]],
            [[0, 11], [4, 0], [5, 8], [0, 0]],  # z maps to UNK in base idx, b maps to UNK in copy idx
        ])

    def test_embed_seq_batch(self, multi_vocab_indices, token_embedder):
        embeds = token_embedder.embed_seq_batch(multi_vocab_indices)

        assert_tensor_equal(embeds.values, [
            [[10, 11, 20, 21], [8, 9, 18, 19], [10, 11, 20, 21], [10, 11, 20, 21]],
            [[0, 1, 22, 23], [8, 9, 0, 1], [10, 11, 16, 17], [0, 1, 0, 1]],
        ])

        assert_tensor_equal(embeds.mask, multi_vocab_indices.mask)

    def test_embed_tokens(self, token_embedder):
        assert_tensor_equal(token_embedder.embed_tokens(['c', 'z']), [
            [10, 11, 20, 21],
            [0, 1, 22, 23]
        ])

        assert_tensor_equal(token_embedder.embed_tokens(['b', 'b']), [
            [8, 9, 18, 19],
            [8, 9, 0, 1]
        ])

        assert_tensor_equal(token_embedder.embed_tokens(['c', 'c']), [
            [10, 11, 20, 21],
            [10, 11, 16, 17]
        ])

        assert_tensor_equal(token_embedder.embed_tokens(['c', WordVocab.STOP]), [
            [10, 11, 20, 21],
            [4, 5, 0, 1]
        ])
