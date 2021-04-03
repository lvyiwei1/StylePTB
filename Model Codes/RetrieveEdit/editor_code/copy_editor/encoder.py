from collections import namedtuple

import torch
from torch.nn import LSTMCell, Module, Linear

from editor_code.copy_editor.vocab import base_plus_copy_indices
from gtd.ml.torch.seq_batch import SequenceBatch
from gtd.ml.torch.source_encoder import MultiLayerSourceEncoder
from gtd.ml.torch.utils import NamedTupleLike
from gtd.ml.torch.utils import GPUVariable
import numpy as np

EncoderInput = namedtuple('EncoderInput', ['input_words', 'output_words', 'token_embedder'])
"""
Args:
    input_words (list[MultiVocabIndices])
    output_words (list[MultiVocabIndices])
    token_embedder (DynamicMultiVocabTokenEmbedder)
"""


class EncoderOutput(namedtuple('EncoderOutput', ['input_embeds', 'agenda', 'token_embedder']), NamedTupleLike):
    pass
"""
Args:
    input_embeds (list[SequenceBatch]): list of SequenceBatch elements of shape (batch_size, seq_length, hidden_size)
    agenda (Variable): of shape (batch_size, agenda_dim) - VAE latent variable for reconstructing target sequence
    token_embedder (DynamicMultiVocabTokenEmbedder)
"""

class Encoder(Module):
    def __init__(self, word_dim, agenda_dim, hidden_dim, num_layers, num_inputs, dropout_prob, use_vae, kappa=0, use_target=True):
        """Construct Encoder.

        Args:
            word_dim (int)
            agenda_dim (int)
            hidden_dim (int)
            num_layers (int)
            num_inputs (int)
        """
        super(Encoder, self).__init__()

        self.agenda_dim = agenda_dim
        self.source_encoder = MultiLayerSourceEncoder(2 * word_dim, hidden_dim, num_layers, dropout_prob=dropout_prob,
                                                      rnn_cell_factory=LSTMCell)
        self.target_encoder = MultiLayerSourceEncoder(2 * word_dim, hidden_dim, num_layers, dropout_prob=0.0,
                                                      rnn_cell_factory=LSTMCell)
        self.agenda_maker = AgendaMaker(self.source_encoder.hidden_dim * num_inputs, self.agenda_dim)
        self.output_agenda_maker = AgendaMaker(self.source_encoder.hidden_dim, self.agenda_dim)
        self.use_vae = use_vae
        self.vae_wrap = VMFVAEWrapper(kappa)
        self.use_target = use_target

    def preprocess(self, input_batches, output_seqs, token_embedder, volatile=False):
        """Preprocess.

        Args:
            input_batches (list[list[list[unicode]]]): a batch of input sequence lists
                Each sequence list has one sequence per "input channel"
            output_seqs (list[list[unicode]]): a batch of output sequences (targets)
            token_embedder (DynamicMultiVocabTokenEmbedder)
            volatile (bool): whether to make Variables volatile (don't track gradients)

        Returns:
            EncoderInput
        """

        dynamic_vocabs = token_embedder.dynamic_vocabs
        base_vocab = token_embedder.base_vocab
        indices_list = []
        for channel_seqs in zip(*input_batches):
            # channel_seqs is a batch of sequences, where all the sequences come from one "input channel"
            multi_vocab_indices = base_plus_copy_indices(list(channel_seqs), dynamic_vocabs, base_vocab, volatile=volatile)
            indices_list.append(multi_vocab_indices)

        output_indices = base_plus_copy_indices(output_seqs, dynamic_vocabs, base_vocab, volatile=volatile)

        return EncoderInput(indices_list, output_indices, token_embedder)


    def make_embedding(self, encoder_input, words_list, encoder):
        """Encoder for a single `channel'
        """
        channel_word_embeds = encoder_input.token_embedder.embed_seq_batch(words_list)
        source_encoder_output = encoder(channel_word_embeds.split())

        channel_embeds_list = source_encoder_output.combined_states
        channel_embeds = SequenceBatch.cat(channel_embeds_list)

        # the final hidden states in both the forward and backward direction, concatenated
        channel_embeds_final = torch.cat(source_encoder_output.final_states, 1)  # (batch_size, hidden_dim)
        return channel_embeds, channel_embeds_final


    def target_out(self, encoder_input):
        output_embeds, output_embeds_final = self.make_embedding(encoder_input, encoder_input.output_words,
                                                                 self.target_encoder)

        return self.output_agenda_maker(output_embeds_final)

    def ctx_code_out(self, encoder_input):
        all_channel_embeds = []
        all_channel_embeds_final = []

        for channel_words in encoder_input.input_words:
            channel_embeds, channel_embeds_final = self.make_embedding(encoder_input, channel_words,
                                                                       self.source_encoder)
            all_channel_embeds.append(channel_embeds)
            all_channel_embeds_final.append(channel_embeds_final)

        input_embeds_final = torch.cat(all_channel_embeds_final, 1)  # (batch_size, hidden_dim * num_channels)
        context_agenda = self.agenda_maker(input_embeds_final)
        return context_agenda, all_channel_embeds



    def forward(self, encoder_input, train_mode=True):
        """Encode.

        Args:
            encoder_input (EncoderInput)

        Returns:
            EncoderOutput, cost (0 in this case)
        """
        context_agenda, all_channel_embeds = self.ctx_code_out(encoder_input)

        if self.use_vae and train_mode:
            if self.use_target:
                target_agenda = self.target_out(encoder_input)
                vae_agenda, vae_loss = self.vae_wrap(context_agenda+target_agenda, True)
            else:
                vae_agenda, vae_loss = self.vae_wrap(context_agenda, True)
        else:
            vae_agenda = context_agenda / torch.sqrt(torch.sum(context_agenda**2.0, dim=1)).expand_as(context_agenda)
            vae_loss = GPUVariable(torch.zeros(1))

        return EncoderOutput(all_channel_embeds, vae_agenda, encoder_input.token_embedder), vae_loss


class AgendaMaker(Module):
    def __init__(self, source_dim, agenda_dim):
        super(AgendaMaker, self).__init__()
        self.linear = Linear(source_dim, agenda_dim)

    def forward(self, source_embed):
        """Create agenda vector from source text embedding and edit embedding.

        Args:
            source_embed (Variable): of shape (batch_size, source_dim)

        Returns:
            agenda (Variable): of shape (batch_size, agenda_dim)
        """
        return self.linear(source_embed)

class GaussianVAEWrapper(Module):
    def __init__(self, vae_wt):
        super(GaussianVAEWrapper, self).__init__()
        self.vae_wt = vae_wt

    def kl_penalty(self, agenda):
        """
        Computes KL penalty given encoder output
        """
        batch_size, agenda_dim = agenda.size()
        return self.vae_wt * 0.5 * torch.sum(torch.pow(agenda, 2)) / batch_size

    def forward(self, source_embed, add_noise=True):
        means = torch.zeros(source_embed.size())
        std = torch.ones(source_embed.size())
        noise = GPUVariable(torch.normal(means=means, std=std))  # unit Gaussian
        if add_noise:
            return source_embed + noise, self.kl_penalty(source_embed)
        else:
            return source_embed, 0

class VMFVAEWrapper(Module):
    def __init__(self, kappa):
        super(VMFVAEWrapper, self).__init__()
        self.kappa = kappa

    def kl_penalty(self, agenda):
        # igoring this for simplicity since we don't need the KL penalty.
        batch_size, id_dim = agenda.size()
        return GPUVariable(torch.zeros(1))

    def sample_vMF(self, mu, kappa):
        """vMF sampler in pytorch.

        Args:
            mu (Tensor): of shape (batch_size, 2*word_dim)
            kappa (Float): controls dispersion. kappa of inf is no dispersion.
        """
        batch_size, id_dim = mu.size()
        result_list = []
        for i in range(batch_size):
            munorm = mu[i].norm().expand(id_dim)
            # sample offset from center (on sphere) with spread kappa
            w = self._sample_weight(kappa, id_dim)
            wtorch = GPUVariable(w * torch.ones(id_dim))

            # sample a point v on the unit sphere that's orthogonal to mu
            v = self._sample_orthonormal_to(mu[i] / munorm, id_dim)

            # compute new point
            scale_factr = torch.sqrt(GPUVariable(torch.ones(id_dim)) - torch.pow(wtorch, 2))
            orth_term = v * scale_factr
            muscale = mu[i] * wtorch / munorm
            sampled_vec = (orth_term + muscale)
            result_list.append(sampled_vec)
        return torch.stack(result_list, 0)

    def renormalize_norm(self, mu):
        """

        Args:
            mu (Tensor): of shape (batch_size, 2*word_dim)
        """
        batch_size, id_dim = mu.size()
        result_list = []
        for i in range(batch_size):
            munorm = mu[i].norm().expand(id_dim)
            sampled_vec = mu[i] / munorm
            result_list.append(sampled_vec)

        return torch.stack(result_list, 0)


    def _sample_weight(self, kappa, dim):
        """Rejection sampling scheme for sampling distance from center on
        surface of the sphere.
        """
        dim = dim - 1  # since S^{n-1}
        b = dim / (np.sqrt(4. * kappa ** 2 + dim ** 2) + 2 * kappa)  # b= 1/(sqrt(4.* kdiv**2 + 1) + 2 * kdiv)
        x = (1. - b) / (1. + b)
        c = kappa * x + dim * np.log(1 - x ** 2)  # dim * (kdiv *x + np.log(1-x**2))

        while True:
            z = np.random.beta(dim / 2., dim / 2.)  # concentrates towards 0.5 as d-> inf
            w = (1. - (1. + b) * z) / (1. - (1. - b) * z)
            u = np.random.uniform(low=0, high=1)
            if kappa * w + dim * np.log(1. - x * w) - c >= np.log(
                    u):  # thresh is dim *(kdiv * (w-x) + log(1-x*w) -log(1-x**2))
                return w

    def _sample_orthonormal_to(self, mu, dim):
        """Sample point on sphere orthogonal to mu.
        """
        v = GPUVariable(torch.randn(dim))
        rescale_value = mu.dot(v) / mu.norm()
        proj_mu_v = mu * rescale_value.expand(dim)
        ortho = v - proj_mu_v
        ortho_norm = torch.norm(ortho)
        return ortho / ortho_norm.expand_as(ortho)

    def forward(self, source_embed, add_noise=True):
        if add_noise:
            return self.sample_vMF(source_embed, self.kappa), self.kl_penalty(source_embed)
        else:
            return self.renormalize_norm(source_embed), 0

