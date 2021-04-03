from itertools import izip

import numpy as np
import torch
from torch.nn import Module

from gtd.ml.torch.utils import GPUVariable
from editor_code.copy_editor.utils import STOPWORDS
from gtd.ml.torch.decoder import ExtensionProbsModifier, BeamDuplicatable
from gtd.ml.torch.seq_batch import SequenceBatch
from gtd.ml.torch.utils import assert_tensor_equal
from gtd.ml.vocab import Vocab, CasedWordVocab, SimpleEmbeddings, emulate_distribution
from gtd.utils import flatten


class HardCopyVocab(CasedWordVocab):
    def __init__(self, base_tokens, num_copy_tokens):
        """Vocab containing hard copy tokens.
        
        hard copy tokens are appended to the end of the index.
        
        Args:
            base_tokens (list[unicode]): base tokens, NOT including CasedWordVocab.SPECIAL_TOKENS or copy tokens
            num_copy_tokens (int): number of hard copy tokens to create
        """
        tokens = list(CasedWordVocab.SPECIAL_TOKENS)  # start with special tokens
        tokens.extend(base_tokens)  # add base tokens

        # add copy tokens
        copy_tokens = [self.copy_index_to_token(i) for i in xrange(num_copy_tokens)]
        tokens.extend(copy_tokens)

        # create CasedWordVocab. Note that case is ignored.
        super(HardCopyVocab, self).__init__(tokens)

        self._copy_tokens = copy_tokens

    @property
    def copy_tokens(self):
        return self._copy_tokens

    @classmethod
    def copy_token_to_index(cls, copy_token):
        """Return the copy index of the copy token."""
        return int(copy_token[5:-1])

    @classmethod
    def copy_index_to_token(cls, copy_idx):
        return '<copy{}>'.format(copy_idx)


# TODO(kelvin): test this
def load_embeddings(file_path, word_dim, vocab_size, num_copy_tokens):
    special_tokens = CasedWordVocab.SPECIAL_TOKENS

    base_embeds = SimpleEmbeddings.from_file(file_path, word_dim, vocab_size)
    _, embed_dim = base_embeds.array.shape

    def sample_embeds(num_embeds, seed):
        shape = (num_embeds, embed_dim)
        return emulate_distribution(shape, base_embeds.array, seed=seed)

    special_tokens_array = sample_embeds(len(special_tokens), seed=0)
    copy_tokens_array = sample_embeds(num_copy_tokens, seed=1)  # different seed to have different val

    # copy tokens are appended at the end
    new_array = np.concatenate((special_tokens_array, base_embeds.array, copy_tokens_array), axis=0)
    new_vocab = HardCopyVocab(base_embeds.vocab.tokens, num_copy_tokens)

    # check that tokens come in the order that we assumed
    correct_tokens = list(special_tokens)  # special tokens first
    correct_tokens.extend(base_embeds.vocab.tokens)  # then base tokens
    correct_tokens.extend('<copy{}>'.format(i) for i in xrange(num_copy_tokens))  # copy tokens last
    assert new_vocab.tokens == correct_tokens

    return SimpleEmbeddings(new_array, new_vocab)


class HardCopyDynamicVocab(Vocab):
    def __init__(self, hard_copy_vocab, source_tokens, token_lengths):
        """HardCopyDynamicVocab.
        
        NOTE: HardCopyDynamicVocab is blind to casing.
        
        Args:
            hard_copy_vocab (HardCopyVocab)
            source_tokens (list[list[unicode]]) -- outer list is over different input channels, inner list is a sentence
        """
        num_channels = len(source_tokens)

        tok_cum = [0]+np.cumsum(token_lengths).tolist()
        tokens_by_channel = [hard_copy_vocab.copy_tokens[tok_cum[i]:tok_cum[i+1]] for i in range(num_channels)]
        self.tokens_by_channel = tokens_by_channel

        flatten = lambda l: [item for sublist in l for item in sublist]

        copy_words = [sentence_tokens + [hard_copy_vocab.STOP] for sentence_tokens in source_tokens]
        copy_pairs = flatten([zip(copy_words[i], tokens_by_channel[i]) for i in range(num_channels)])
        # pair each copy word with a copy token
        # note that zip only zips up to the shorter of the two (desired behavior)

        #old snippet below
        #copy_words = [self.get_copy_words(sentence_tokens, hard_copy_vocab) for sentence_tokens in source_tokens]
        #copy_pairs = zip(copy_words, hard_copy_vocab.copy_tokens)
        self.word_to_copy_token = {w: t for w, t in copy_pairs[::-1]}
        self.copy_token_to_word = {t: w for w, t in copy_pairs[::-1]}

        # map from word to index
        self.base_vocab = hard_copy_vocab

    @classmethod
    def get_copy_words(cls, words, hard_copy_vocab):
        """Unique and lower-cased words from `words`, sorted by their appearance in `words`."""
        #words = [w.lower() for w in words]
        words = [w for w in words]
        seen = set()
        copy_words = []
        for w in words:
            if w not in seen:
                # only add words that haven't already been seen
                copy_words.append(w)
                seen.add(w)
        return copy_words

    def word2index(self, w):
        # NOTE: we do NOT replace the word with a copy token, even if we have assigned it a copy token
        return self.base_vocab.word2index(w)

    def index2word(self, i):
        w = self.base_vocab.index2word(i)
        # replace copy token with word
        return self.copy_token_to_word.get(w, w)


class LexicalWhitelister(ExtensionProbsModifier):
    def __init__(self, whitelists, vocab, word_to_forms, always_allowed=None):
        """Modify extension probs such that only words appearing in the whitelist (or have a variant in the whitelist) are allowed.
        
        Args:
            whitelists (list[list[unicode]]): a batch of whitelists, one per example.
                Each whitelist is a list of permitted words.
            vocab (HardCopyVocab)
            word_to_forms (Callable[unicode, list[unicode]): a function mapping words to forms
            always_allowed (list[unicode]): a list of words that is allowed in any example
        """
        # importantly, vocab should NOT be a HardCopyDynamicVocab
        assert isinstance(vocab, HardCopyVocab)

        if always_allowed is None:
            # always allow STOP, copy tokens, and various stop words
            always_allowed = []
            always_allowed.append(vocab.STOP)
            always_allowed.extend(vocab.copy_tokens)
            always_allowed.extend(STOPWORDS)  # all lower case

        # so far, these should all be unique
        always_allowed_indices = [vocab.word2index(w) for w in always_allowed if w in vocab]

        # precompute actual allowed words
        all_whitelist_words = set(flatten(whitelists))
        word_to_form_indices = self._word_to_form_indices(all_whitelist_words, vocab, word_to_forms)

        allowed_indices = []
        for whitelist in whitelists:
            indices = self._whitelist_to_indices(whitelist, word_to_form_indices, always_allowed_indices)
            allowed_indices.append(indices)
            # Returned indices may contain some duplicates.

        self.allowed_indices = allowed_indices

    @classmethod
    def _word_to_form_indices(cls, words, vocab, word_to_forms):
        """Return a map from each word to a list of word indices (indices of the word's forms).
        
        If a form is not in the vocab, it is NOT included.
        """
        word_to_form_indices = {}
        for word in words:
            if word not in word_to_form_indices:
                forms = word_to_forms(word)
                indices = [vocab.word2index(form) for form in forms if form in vocab]  # avoid including UNK
                word_to_form_indices[word] = indices
        return word_to_form_indices

    @classmethod
    def _whitelist_to_indices(cls, whitelist, word_to_form_indices, always_allowed_indices):
        """Map the whitelist to allowed indices in the vocab.
        
        Always include the `always_allowed_indices`
        
        Returned indices may contain some duplicates.
        
        Args:
            whitelist (list[unicode])
            word_to_form_indices (dict[unicode, list[int])
            always_allowed_indices (list[int])

        Returns:
            list[int]
        """
        allowed_indices = []
        for word in whitelist:
            if word in word_to_form_indices:
                allowed_indices.extend(word_to_form_indices[word])
        allowed_indices.extend(always_allowed_indices)
        return allowed_indices

    def modify(self, extension_probs, rnn_state, states):
        modified_extension_probs = np.zeros(extension_probs.shape, dtype=extension_probs.dtype)

        for i, indices in enumerate(self.allowed_indices):
            for j in indices:
                modified_extension_probs[i, j] = extension_probs[i, j]

        return modified_extension_probs


class MultiVocabIndices(SequenceBatch):
    """A marker class, indicating that SequenceBatch has the following dimensions:
    
    Attributes:
        values (Variable): has shape (batch_size, seq_length, num_vocabs)
        mask (Variable): has shape (batch_size, seq_length)
    """
    pass


class DynamicMultiVocabTokenEmbedder(Module, BeamDuplicatable):
    def __init__(self, token_embedder, dynamic_vocabs, base_vocab):
        """Embed each token as the concatenation of embeddings.
        
        Maps each token to a tuple of indices, then looks up embeddings for each of those indices
        and concatenates them.

        Args:
            token_embedder (TokenEmbedder)
            dynamic_vocabs (list[HardCopyDynamicVocab]): a batch of vocabs, one per example
            base_vocab (HardCopyVocab)
        """
        super(DynamicMultiVocabTokenEmbedder, self).__init__()
        self.base_embedder = token_embedder
        self.dynamic_vocabs = dynamic_vocabs
        self.base_vocab = base_vocab

    def beam_duplicate(self, beam_size):
        duplicated_dynamic_vocabs = flatten([dyna_vocab] * beam_size for dyna_vocab in self.dynamic_vocabs)
        return DynamicMultiVocabTokenEmbedder(self.base_embedder, duplicated_dynamic_vocabs, self.base_vocab)

    @property
    def embeds(self):
        return self.base_embedder.embeds

    def embed_indices(self, indices):
        raise NotImplementedError

    def embed_seq_batch(self, multi_vocab_indices):
        """Embed elements of a SequenceBatch.

        Args:
            multi_vocab_indices (MultiVocabIndices)

        Returns:
            SequenceBatch: with values of shape (batch_size, seq_length, base_embed_dim * num_vocabs)
        """
        # has shape (batch_size, seq_length, num_vocabs, base_embed_dim)
        base_embeds = self.base_embedder.embed_indices(multi_vocab_indices.values)
        # a list where each element is (batch_size, seq_length, 1, base_embed_dim)
        base_embeds_list = torch.split(base_embeds, 1, 2)
        # a list where each element is (batch_size, seq_length, base_embed_dim)
        base_embeds_list = [b_embeds.squeeze(2) for b_embeds in base_embeds_list]
        cat_embeds = torch.cat(base_embeds_list, 2)  # (batch_size, seq_length, base_embed_dim * num_vocabs)
        return SequenceBatch(cat_embeds, multi_vocab_indices.mask)

    def embed_tokens(self, tokens):
        """Embed list of tokens.
        
        NOTE:
            This function is called at test-time decoding.
            For proper functioning, it MUST match the train-time behavior of:
                self.embed_seq_batch
                HardCopyTrainDecoderInput._base_plus_copy_indices

        Args:
            tokens (list[unicode])

        Returns:
            embeds (Variable[FloatTensor]): of shape (len(tokens), base_embed_dim * num_vocabs)
        """
        dynamic_vocabs = self.dynamic_vocabs
        base_vocab = self.base_vocab
        unk = base_vocab.UNK

        # TODO(kelvin): double-check if this is right
        # if it is a copy token, convert it back to a word
        # if it is not a copy token, leave it alone
        # if the copy token has no corresponding word, leave it alone
        base_tokens = [dyna_vocab.copy_token_to_word.get(w, w) for w, dyna_vocab in izip(tokens, dynamic_vocabs)]

        # convert words to copy tokens
        copy_tokens = [dyna_vocab.word_to_copy_token.get(w, unk) for w, dyna_vocab in izip(base_tokens, dynamic_vocabs)]

        # if it is a copy token (not in word_to_copy) leave it alone
        # if it is a word (ie in word_to_copy) then get the copy index, and use its vocab

        # both have shape (batch_size, base_embed_dim)
        base_embeds = self.base_embedder.embed_tokens(base_tokens)
        copy_embeds = self.base_embedder.embed_tokens(copy_tokens)

        return torch.cat([base_embeds, copy_embeds], 1)


def base_plus_copy_indices(words, dynamic_vocabs, base_vocab, volatile=False):
    """Compute base + copy indices.
    
    Args:
        words (list[list[unicode]])
        dynamic_vocabs (list[HardCopyDynamicVocab])
        base_vocab (HardCopyVocab)
        volatile (bool)

    Returns:
        MultiVocabIndices
    """
    unk = base_vocab.UNK
    copy_seqs = []
    for seq, dyna_vocab in izip(words, dynamic_vocabs):
        word_to_copy = dyna_vocab.word_to_copy_token
        normal_copy_seq = []
        for w in seq:
            normal_copy_seq.append(word_to_copy.get(w, unk))
        copy_seqs.append(normal_copy_seq)

    # each SeqBatch.values has shape (batch_size, seq_length)
    base_indices = SequenceBatch.from_sequences(words, base_vocab, volatile=volatile)
    copy_indices = SequenceBatch.from_sequences(copy_seqs, base_vocab, volatile=volatile)

    assert_tensor_equal(base_indices.mask, copy_indices.mask)

    # has shape (batch_size, seq_length, 2)
    concat_values = torch.stack([base_indices.values, copy_indices.values], 2)

    return MultiVocabIndices(concat_values, base_indices.mask)