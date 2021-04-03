import numpy as np
import pytest
import torch

from gtd.ml.torch.token_embedder import TokenEmbedder
from gtd.ml.torch.utils import GPUVariable
from gtd.ml.torch.utils import assert_tensor_equal
from gtd.ml.vocab import SimpleVocab
from gtd.utils import Bunch


class TestTokenEmbedder(object):
    @pytest.fixture(params=[True, False])
    def embedder(self, request):
        vocab = SimpleVocab(['<unk>', '<start>', '<stop>'] + ['a', 'b', 'c'])
        arr = np.eye(len(vocab), dtype=np.float32)
        word_embeddings = Bunch(vocab=vocab, array=arr)
        return TokenEmbedder(word_embeddings, trainable=request.param)

    def test_embed_indices(self, embedder):
        indices = GPUVariable(torch.LongTensor([
            [0, 1],
            [2, 2],
            [4, 5],
        ]))

        embeds = embedder.embed_indices(indices)

        assert_tensor_equal(embeds, [
            [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]],
            [[0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0]],
            [[0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]],
        ])

        indices = GPUVariable(torch.LongTensor([
            [[0, 1], [1, 0]],
            [[2, 2], [3, 2]],
        ]))

        embeds = embedder.embed_indices(indices)
        assert_tensor_equal(embeds, [
            [[[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]], [[0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]]],
            [[[0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0]], [[0, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0]]],
        ])

    def test_embed_tokens(self, embedder):
        tokens = ['b', 'c', 'c']
        embeds = embedder.embed_tokens(tokens)

        assert_tensor_equal(embeds, [
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 1],
        ])