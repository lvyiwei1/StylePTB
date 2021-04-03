import pytest
import numpy as np

from editor_code.copy_editor.editor import HardCopyTrainDecoderInput
from editor_code.copy_editor.vocab import HardCopyVocab, HardCopyDynamicVocab, LexicalWhitelister
from gtd.ml.torch.utils import assert_tensor_equal
from gtd.ml.vocab import WordVocab


class TestHardCopyVocab(object):

    @pytest.fixture
    def vocab(self):
        tokens = 'apple CAT Bat'.split()
        return HardCopyVocab(tokens, 3)

    def test_tokens(self, vocab):
        assert vocab.tokens == [vocab.UNK, vocab.START, vocab.STOP, 'apple', 'cat', 'bat', '<copy0>', '<copy1>', '<copy2>']

    def test_copy_tokens(self, vocab):
        assert vocab.copy_tokens == ['<copy0>', '<copy1>', '<copy2>']


class TestHardCopyDynamicVocab(object):
    @pytest.fixture
    def base_vocab(self):
        tokens = 'apple CAT Bat'.split()
        return HardCopyVocab(tokens, 5)

    @pytest.fixture
    def vocab(self, base_vocab):
        return HardCopyDynamicVocab(base_vocab, 'apple The bat is the BEST time ever'.split())

    def test_too_many_copy_words(self, vocab):
        # only has room for 5 copy words
        # but there are too many
        # will only keep the first 5 that appear
        assert vocab.copy_token_to_word == {
            '<copy0>': 'apple',
            '<copy1>': 'the',
            '<copy2>': 'bat',
            '<copy3>': 'is',
            '<copy4>': 'best',
        }

    def test_too_many_copy_tokens(self, base_vocab):
        vocab = HardCopyDynamicVocab(base_vocab, 'The'.split())
        # should only use one copy token
        assert vocab.word_to_copy_token == {
            'the': '<copy0>',
        }

    def test_word2index(self, vocab):
        unk_idx = vocab.word2index(WordVocab.UNK)
        assert unk_idx == 0
        assert vocab.word2index('what') == unk_idx  # what is neither base nor copy, get assigned UNK
        assert vocab.word2index('time') == unk_idx  # this was in source, but didn't get assigned a copy token
        assert vocab.word2index('The') == unk_idx  # this has a copy token, but we are still assigning unk
        assert vocab.word2index('Apple') == 3  # plain base word

    def test_index2word(self, vocab):
        unk_idx = vocab.word2index(WordVocab.UNK)
        assert unk_idx == 0
        assert vocab.index2word(unk_idx) == '<unk>'  # unk should still be unk
        assert vocab.index2word(4) == 'cat'  # base word should still be the same
        assert vocab.index2word(8) == 'bat'  # should get word back, not copy token
        assert vocab.index2word(7) == 'the'  # should get word back, not copy token


class TestHardCopyTrainDecoderInput(object):
    @pytest.fixture
    def vocab(self):
        tokens = 'apple bat cat'.split()
        return HardCopyVocab(tokens, 3)

    def test(self, vocab):
        pass  # TODO


class TestLemmatizedVocabWhitelist(object):
    @pytest.fixture
    def vocab(self):
        tokens = ['eat', 'ate', 'eaten'] + ['hello', 'you', 'me'] + ['drive', 'drove', 'driven']
        return HardCopyVocab(tokens, num_copy_tokens=3)

    def word_to_forms(self, word):
        if word == 'eat':
            return ['eat', 'ate', 'eaten']
        elif word == 'drive':
            return ['drive', 'drove', 'driven', 'drived']
        else:
            return []

    def test_word_to_form_indices(self, vocab):
        words = ['eat', 'hey', 'drive', 'eat', 'hey']
        assert LexicalWhitelister._word_to_form_indices(words, vocab, self.word_to_forms) == {
            'eat': [3, 4, 5],
            'hey': [],
            'drive': [9, 10, 11],  # 'drived' is out of vocab, and therefore should not get an index
        }

    def test_whitelist_to_indices(self):
        word_to_form_indices = {
            'hi': [1, 2, 3],
            'you': [3, 4, 5],
        }
        extra_allowed_indices = [-1, -2, -3]
        assert LexicalWhitelister._whitelist_to_indices(
            ['you', 'there', 'hi'],
            word_to_form_indices,
            extra_allowed_indices) == [3, 4, 5, 1, 2, 3, -1, -2, -3]

    def test_modify(self, vocab):
        whitelists = [
            ['eat', 'what', 'drive'],  # only eat and drive will have lemmas matched
            ['hello', 'drive'],  # only drive will have lemmas matched
            ['hi'],  # no lemmas will match
        ]

        always_allowed = []
        always_allowed.append(vocab.STOP)
        always_allowed.extend(vocab.copy_tokens)

        modifier = LexicalWhitelister(whitelists, vocab, self.word_to_forms, always_allowed=always_allowed)
        extension_probs = np.reshape(np.arange(0, 45, dtype=np.float32), (3, 15))
        modified = modifier.modify(extension_probs, None, None)

        # STOP token is enabled
        # all the copy tokens are enabled
        assert_tensor_equal(modified, [
            [0, 0,  2,  3, 4, 5, 0, 0, 0, 9,  10, 11, 12, 13, 14],
            [0, 0, 17, 0, 0, 0, 0, 0, 0, 24, 25, 26, 27, 28, 29],
            [0, 0, 32, 0, 0, 0, 0, 0, 0, 0,  0,  0,  42, 43, 44],
        ])