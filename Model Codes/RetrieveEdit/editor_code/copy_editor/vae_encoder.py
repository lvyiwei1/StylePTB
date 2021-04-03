import torch
from torch.nn import Module
from gtd.ml.torch.utils import GPUVariable

from collections import namedtuple

import torch
from torch.nn import LSTMCell, Module, Linear

from editor_code.copy_editor.vocab import base_plus_copy_indices
from gtd.ml.torch.seq_batch import SequenceBatch
from gtd.ml.torch.source_encoder import MultiLayerSourceEncoder
from gtd.ml.torch.utils import NamedTupleLike
from gtd.ml.torch.utils import GPUVariable
import numpy as np

from editor_code.copy_editor.encoder import EncoderInput, EncoderOutput, VMFVAEWrapper, AgendaMaker

class TargetVAEEncoder(Module):
    def __init__(self, word_dim, agenda_dim, hidden_dim, num_layers, num_inputs):
        """Construct Encoder.

        Args:
            word_dim (int)
            agenda_dim (int)
            hidden_dim (int)
            num_layers (int)
            num_inputs (int)
        """
        super(TargetVAEEncoder, self).__init__()

        self.agenda_dim = agenda_dim
        self.target_encoder = MultiLayerSourceEncoder(2 * word_dim, hidden_dim, num_layers, dropout_prob=0.0,
                                                      rnn_cell_factory=LSTMCell)
        self.agenda_maker = AgendaMaker(self.target_encoder.hidden_dim * num_inputs, self.agenda_dim)
        self.output_agenda_maker = AgendaMaker(self.target_encoder.hidden_dim, self.agenda_dim)
        self.vae_wrap = VMFVAEWrapper(500)

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
            multi_vocab_indices = base_plus_copy_indices(list(channel_seqs), dynamic_vocabs, base_vocab,
                                                         volatile=volatile)
            indices_list.append(multi_vocab_indices)

        output_indices = base_plus_copy_indices(output_seqs, dynamic_vocabs, base_vocab, volatile=volatile)
        return EncoderInput(indices_list, output_indices, token_embedder)

    def make_embedding(self, encoder_input, words_list, encoder):
        """Encoder for a single `channel'
        """
        channel_word_embeds = encoder_input.token_embedder.embed_seq_batch(words_list)
        encoder_output = encoder(channel_word_embeds.split())

        channel_embeds_list = encoder_output.combined_states
        channel_embeds = SequenceBatch.cat(channel_embeds_list)

        # the final hidden states in both the forward and backward direction, concatenated
        channel_embeds_final = torch.cat(encoder_output.final_states, 1)  # (batch_size, hidden_dim)
        return channel_embeds, channel_embeds_final

    def ctx_code_out(self, encoder_input):
        all_channel_embeds = []
        all_channel_embeds_final = []

        for channel_words in encoder_input.input_words:
            channel_embeds, channel_embeds_final = self.make_embedding(encoder_input, channel_words,
                                                                       self.target_encoder)
            all_channel_embeds.append(channel_embeds)
            all_channel_embeds_final.append(channel_embeds_final)

        input_embeds_final = torch.cat(all_channel_embeds_final, 1)  # (batch_size, hidden_dim * num_channels)
        context_agenda = self.agenda_maker(input_embeds_final)
        return context_agenda, all_channel_embeds

    def target_out(self, encoder_input):
        output_embeds, output_embeds_final = self.make_embedding(encoder_input, encoder_input.output_words,
                                                                 self.target_encoder)

        return self.output_agenda_maker(output_embeds_final)

    def forward(self, encoder_input):
        """Encode.

        Args:
            encoder_input (EncoderInput)

        Returns:
            EncoderOutput, cost (0 in this case)
        """
        context_agenda, all_channel_embeds = self.ctx_code_out(encoder_input)
        target_agenda = self.target_out(encoder_input)
        vae_agenda, vae_loss = self.vae_wrap(context_agenda, True)

        return EncoderOutput(all_channel_embeds, vae_agenda, encoder_input.token_embedder), vae_loss


