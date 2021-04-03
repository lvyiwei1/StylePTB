from itertools import izip

import numpy as np
import torch
from torch.nn import LSTMCell, Linear, Parameter, Softmax, Dropout

from collections import namedtuple
from gtd.ml.torch.attention import Attention, AttentionOutput
from gtd.ml.torch.decoder_cell import DecoderCell, DecoderCellOutput, RNNState, \
    RNNInput, PredictionBatch
from gtd.ml.torch.recurrent import gated_update, tile_state
from gtd.ml.torch.utils import GPUVariable
from gtd.utils import UnicodeMixin
from gtd.ml.torch.decoder import RNNContextCombiner


class AttentionContextCombiner(RNNContextCombiner):
    def __call__(self, encoder_output, x):
        return AttentionRNNInput(x=x, agenda=encoder_output.agenda, input_embeds=encoder_output.input_embeds,
                                 token_embedder=encoder_output.token_embedder)


class AttentionDecoderCell(DecoderCell):
    def __init__(self, target_token_embedder, input_dim, agenda_dim, decoder_dim, encoder_dim,
                 attn_dim, num_layers, num_inputs, dropout_prob, disable_attention):
        super(AttentionDecoderCell, self).__init__()

        target_dim = target_token_embedder.embed_dim
        self.num_layers = num_layers
        self.num_inputs = num_inputs
        self.disable_attention = disable_attention

        if disable_attention:
            augment_dim = agenda_dim
        else:
            # see definition of `x_augment` in `forward` method
            # we augment the input to each RNN layer with num_inputs attention contexts + the agenda
            augment_dim = encoder_dim * num_inputs + agenda_dim


        self.rnn_cells = []
        for layer in range(num_layers):
            in_dim = input_dim if layer == 0 else decoder_dim  # first layer takes word vectors
            out_dim = decoder_dim
            rnn_cell = LSTMCell(in_dim + augment_dim, out_dim)
            self.add_module('decoder_layer_{}'.format(layer), rnn_cell)
            self.rnn_cells.append(rnn_cell)

        if disable_attention:
            z_dim = decoder_dim
        else:
            # see definition of `z` in `forward` method
            # to predict words, we condition on the hidden state h + num_inputs attention context
            z_dim = decoder_dim + encoder_dim * num_inputs

        # TODO(kelvin): these big params may need regularization
        self.vocab_projection_pos = Linear(z_dim, target_dim)
        self.vocab_projection_neg = Linear(z_dim, target_dim)
        self.relu = torch.nn.ReLU()

        self.h0 = Parameter(torch.zeros(decoder_dim))
        self.c0 = Parameter(torch.zeros(decoder_dim))
        self.vocab_softmax = Softmax()

        self.input_attentions = []
        for i in range(num_inputs):
            attn = Attention(encoder_dim, decoder_dim, attn_dim)
            self.add_module('input_attention_{}'.format(i), attn)
            self.input_attentions.append(attn)

        self.target_token_embedder = target_token_embedder

        self.dropout = Dropout(dropout_prob)

    def initialize(self, batch_size):
        h = tile_state(self.h0, batch_size)
        c = tile_state(self.c0, batch_size)

        # no initial weights, context is just zero vector
        init_attn = lambda attention: AttentionOutput(weights=None,
            context=GPUVariable(torch.zeros(batch_size, attention.memory_dim)),
            logits=None
        )

        return AttentionRNNState([h] * self.num_layers, [c] * self.num_layers,
                                 [init_attn(attn) for attn in self.input_attentions])

    def forward(self, rnn_state, decoder_cell_input, advance):
        dci = decoder_cell_input
        mask = advance

        # this will be concatenated to x at every layer
        # we are conditioning on the attention from the previous time step and the agenda from the encoder
        attn_contexts = torch.cat([attn.context for attn in rnn_state.input_attns], 1)
        if self.disable_attention:
            x_augment = dci.agenda
        else:
            x_augment = torch.cat([attn_contexts, dci.agenda], 1)

        hs, cs = [], []
        x = dci.x  # input word vector
        for layer in range(self.num_layers):
            rnn_cell = self.rnn_cells[layer]
            old_h, old_c = rnn_state.hs[layer], rnn_state.cs[layer]
            rnn_input = torch.cat([x, x_augment], 1)
            h, c = rnn_cell(rnn_input, (old_h, old_c))
            h = gated_update(old_h, h, mask)
            c = gated_update(old_c, c, mask)
            hs.append(h)
            cs.append(c)

            if layer == 0:
                x = h  # no skip connection on the first layer
            else:
                x = x + h

            # Recurrent Neural Network Regularization
            # https://arxiv.org/pdf/1409.2329.pdf
            x = self.dropout(x)
            # note that dropout doesn't touch the recurrent connections
            # only connections going up the layers

        # compute attention using bottom layer
        input_attns = [attn(dci.input_embeds[i], hs[0]) for i, attn in enumerate(self.input_attentions)]
        attn_contexts = torch.cat([attn.context for attn in input_attns], 1)
        if self.disable_attention:
            z = x
        else:
            z = torch.cat([x, attn_contexts], 1)

        # has shape (batch_size, decoder_dim + encoder_dim + input_dim + input_dim)
        vocab_query_pos = self.vocab_projection_pos(z)
        vocab_query_neg = self.vocab_projection_neg(z)
        word_embeds = self.target_token_embedder.embeds
        vocab_logit_pos = self.relu(torch.mm(vocab_query_pos,
                                             word_embeds.t()))  # (batch_size, vocab_size)
        vocab_logit_neg = self.relu(torch.mm(vocab_query_neg,
                                             word_embeds.t()))  # (batch_size, vocab_size)
        vocab_probs = self.vocab_softmax(vocab_logit_pos - vocab_logit_neg)
        # TODO(kelvin): prevent model from putting probability on UNK

        rnn_state = AttentionRNNState(hs, cs, input_attns)

        # DynamicMultiVocabTokenEmbedder
        # NOTE: this is the same token embedder used by the SOURCE encoder
        dynamic_token_embedder = dci.token_embedder
        base_vocab = dynamic_token_embedder.base_vocab
        dynamic_vocabs = dynamic_token_embedder.dynamic_vocabs

        return AttentionDecoderCellOutput(rnn_state, base_vocab=base_vocab,
                                          dynamic_vocabs=dynamic_vocabs, vocab_probs=vocab_probs)

    def rnn_state_type(self):
        return AttentionRNNState

    def rnn_input_type(self):
        return AttentionRNNInput


class AttentionRNNState(namedtuple('AttentionRNNState', ['hs', 'cs', 'input_attns']), RNNState):
    """
    Attributes:
    hs (list[Variable]): a list of the hidden states for each layer of a multi-layer RNN.
        Each Variable has shape (batch_size, hidden_dim).
    cs (list[Variable]): a list of the cell states for each layer of a multi-layer RNN
        Each Variable has shape (batch_size, hidden_dim).
    input_attns (list[AttentionOutput]): one attention for each input channel
    """
    pass


class AttentionRNNInput(namedtuple('AttentionRNNInput', ['x', 'agenda', 'input_embeds', 'token_embedder']), RNNInput):
    """
Attributes:
    x (Variable): of shape (batch_size, word_dim), embedding of word generated at previous time step
    agenda (Variable): of shape (batch_size, agenda_dim)
    input_embeds (list[SequenceBatch]): list of SequenceBatches of shape (batch_size, source_seq_length, hidden_size)
    token_embedder (DynamicMultiVocabTokenEmbedder)
    """
    pass


class AttentionTrace(UnicodeMixin):
    __slots__ = ['name', 'tokens', 'attention_weights']

    def __init__(self, name, tokens, attention_weights):
        """Construct AttentionTrace.

        Args:
            name (unicode): name of attention mechanism
            tokens (list[unicode])
            attention_weights (np.ndarray): a 1D array. May be longer than len(tokens) due to batching.
        """
        assert len(attention_weights.shape) == 1

        # any attention weights exceeding length of tokens should be zero
        for i in range(len(tokens), len(attention_weights)):
            assert attention_weights[i] == 0

        self.name = name
        self.tokens = tokens
        self.attention_weights = attention_weights

    def __unicode__(self):
        total_mass = np.sum(self.attention_weights)
        s = u' '.join(u'{}[{:.2f}]'.format(t, w) for t, w in
                      izip(self.tokens, self.attention_weights))
        return u'{:10}[{:.2f}]: {}'.format(self.name, total_mass, s)


class AttentionDecoderCellOutput(namedtuple('DecoderCellOutput', ['rnn_state', 'base_vocab', 'dynamic_vocabs', 'vocab_probs'])):
    """
    Attributes:
        rnn_state (RNNState)
        vocab (Vocab): just redirects to base_vocab
        base_vocab (HardCopyVocab)
        dynamic_vocabs (list[HardCopyDynamicVocab])
        vocab_probs (Variable): of shape (batch_size, vocab_size)
    """
    @property
    def vocab(self):
        return self.base_vocab

    def loss(self, target_word):
        """Compute loss for this time step.

        Args:
            target_word (Variable): LongTensor of shape (batch_size, 2)
                First dimension is the base word index
                Second dimension is the copy token index

        Returns:
            loss (Variable): of shape (batch_size,)
        """
        target_probs = torch.gather(self.vocab_probs, 1, target_word)

        # mask out probability contributions from UNK
        base_vocab = self.base_vocab
        unk_idx = base_vocab.word2index(base_vocab.UNK)
        not_unk = (target_word != unk_idx).float()  # (batch_size, 2)
        target_probs = target_probs * not_unk  # all UNK targets will be multiplied by zero

        # add together base word prob + copy token prob
        #TODO: stochastically force copy
        target_prob = torch.sum(target_probs, 1).squeeze(1)  # (batch_size,)
        is_unk = 1 - not_unk
        both_unk = torch.prod(is_unk, 1).squeeze(1)  # (batch_size,)
        assert len(target_prob.size()) == 1

        # NOTE: when both targets are UNK, target_prob will be 0
        # this is problematic, because torch.log(0) = -inf
        # hence, artificially set target_prob = 1
        # we will mask these out from the loss below

        target_prob = target_prob + both_unk

        loss = -torch.log(target_prob + 1e-45)  # negative log-likelihood
        # added 1e-45 to prevent loss from being -infinity in the case where probs is close to 0

        loss = loss * (1 - both_unk)  # mask out losses where both targets are UNK

        return loss

    @property
    def predictions(self):
        """Return a PredictionBatch.

        Returns:
            PredictionBatch
        """
        # make a copy and convert to numpy
        vocab_probs_np = self.vocab_probs.data.cpu().numpy()

        # if a word is in the base vocab, transfer probability from copy token to word
        # otherwise, just leave probability on copy token
        self._transfer_copy_probs(vocab_probs_np, self.dynamic_vocabs, self.base_vocab)

        return PredictionBatch(vocab_probs_np, self.base_vocab)

    @classmethod
    def _transfer_copy_probs(cls, vocab_probs, dynamic_vocabs, base_vocab):
        """Transfer probability from copy token to the corresponding word.
        
        NOTE: modifies vocab_probs in place.
        
        Only transfer the probability if the corresponding word is in the base vocab.
        Otherwise, just leave the probability on the copy token.
        
        Args:
            vocab_probs (np.array): of shape (batch_size, base_vocab_size)
            dynamic_vocabs (list[HardCopyDynamicVocab]): batch of vocabs, one per example
            base_vocab (HardCopyVocab)

        Returns:
            vocab_probs (np.array): modified probs
        """

        for i, dyna_vocab in enumerate(dynamic_vocabs):
            for word, copy_token in dyna_vocab.word_to_copy_token.iteritems():
                # TODO: remove after verifying this works .. Disble copy token replacement..
                #if word is base_vocab.STOP:
                if word in base_vocab:
                    word_idx = base_vocab.word2index(word)
                    copy_idx = base_vocab.word2index(copy_token)
                    vocab_probs[i, word_idx] += vocab_probs[i, copy_idx]
                    vocab_probs[i, copy_idx] = 0  # remove prob from copy
                #elif word in base_vocab:
                #    word_idx = base_vocab.word2index(word)
                #    copy_idx = base_vocab.word2index(copy_token)
                #    vocab_probs[i, word_idx] = 0
                #    vocab_probs[i, copy_idx] += vocab_probs[i, word_idx]


