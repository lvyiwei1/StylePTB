import itertools
from collections import namedtuple
from itertools import izip

import numpy as np
import torch
from nltk import word_tokenize
from torch.nn import Module

from editor_code.copy_editor.attention_decoder import AttentionContextCombiner
from editor_code.copy_editor.vocab import HardCopyVocab, HardCopyDynamicVocab, LexicalWhitelister, \
    DynamicMultiVocabTokenEmbedder, base_plus_copy_indices
from gtd.chrono import verboserate
from gtd.log import indent
from gtd.ml.torch.decoder import TrainDecoder, BeamDecoder, TrainDecoderInput
from gtd.text import word_to_forms
from gtd.utils import chunks, UnicodeMixin, flatten
from gtd.ml.torch.seq_batch import SequenceBatch
from editor_code.copy_editor.editor import EditExample, decoder_inputs_and_outputs, HardCopyTrainDecoderInput, EditorInput, EditTrace, LossTrace
from annoy import AnnoyIndex

class VAERetriever(Module):
    def __init__(self, base_source_token_embedder, encoder, decoder_cell, copy_lens):
        super(VAERetriever, self).__init__()
        self.encoder = encoder
        context_combiner = AttentionContextCombiner()
        self.train_decoder = TrainDecoder(decoder_cell, context_combiner)
        self.test_decoder_beam = BeamDecoder(decoder_cell, context_combiner)
        self.base_vocab = base_source_token_embedder.vocab
        self.base_source_token_embedder = base_source_token_embedder
        self.copy_lens = copy_lens

    @classmethod
    def _batch_editor_examples(cls, examples):
        batch = lambda attr: [getattr(ex, attr) for ex in examples]
        input_words = batch('input_words')
        target_words = batch('target_words')
        return input_words, target_words

    ####
    # Retriever code
    def batch_embed(self, exes, train_mode=True):
        ret_list = []
        for batch in chunks(exes, 128):
            encin = self.encode(batch, train_mode).data.cpu().numpy()
            for vec in encin:
                ret_list.append(vec)
        return ret_list

    def make_lsh(self, veclist):
        """
        :param veclist: list of vectors to be indexed
        :return: an annoy LSH index structure.
        """
        lshind = AnnoyIndex(len(veclist[0]), metric='angular')
        for num, vec in enumerate(veclist):
            lshind.add_item(num, vec)
        lshind.build(10)
        return lshind

    def encode(self, in_data, train_mode=True):
        """
        :param in_data: sequence of edit examples - only inputs are encoded!
        :return: batch of agenda vectors of size (batch_size, agenda_dim)
        """
        encoder_input = self.preprocess(in_data).encoder_input
        encoder_output, enc_loss = self.encoder(encoder_input, train_mode=train_mode)
        return encoder_output.agenda

    def ret_lsh(self, veclist, lsh, topk=100, startat=0):
        return [lsh.get_nns_by_vector(vec, topk+startat)[startat:] for vec in veclist]

    def ret_idx(self, in_data, lsh, train_mode=True):
        """
        :param in_data:
        :param lsh:
        :return:
        """
        encoded = self.encode(in_data, train_mode=train_mode)  # batch x agenda_size
        encoded_np = encoded.data.cpu().numpy()
        topk_ret = self.ret_lsh(encoded_np, lsh)
        return topk_ret

    def ret_and_make_ex(self, input, lsh, ex_list, startat, train_mode=True):
        ret_list = []
        for batch in chunks(input, 128):
            idxlist = self.ret_idx(batch, lsh, train_mode=train_mode)
            ret_tmp = [ex_list[idx[startat]] for idx in idxlist]
            ret_list.extend(ret_tmp)
        return self.make_editexamples(ret_list, input)

    def make_editexamples(self, proto_list, edit_list):
        example_list = []
        for i in range(len(proto_list)):
            el = EditExample(edit_list[i].input_words + proto_list[i].input_words + [proto_list[i].target_words],
                             edit_list[i].target_words)
            example_list.append(el)
        return example_list

    ####
    # Editor code (identical to editor)

    def preprocess(self, examples, volatile=False):
        """Preprocess a batch of EditExamples, converting them into arrays.

        Args:
            examples (list[EditExample])

        Returns:
            EditorInput
        """
        input_words, target_words = self._batch_editor_examples(examples)
        dynamic_vocabs = self._compute_dynamic_vocabs(input_words, self.base_vocab)

        dynamic_token_embedder = DynamicMultiVocabTokenEmbedder(self.base_source_token_embedder,
                                                                dynamic_vocabs, self.base_vocab)

        # WARNING:
        # Note that we currently use the same token embedder for both inputs to the encoder and
        # inputs to the decoder. In the future, we may use a different embedder for the decoder.

        encoder_input = self.encoder.preprocess(input_words, target_words, dynamic_token_embedder, volatile=volatile)
        train_decoder_input = HardCopyTrainDecoderInput(target_words, dynamic_vocabs, self.base_vocab)
        return EditorInput(encoder_input, train_decoder_input)

    def _compute_dynamic_vocabs(self, input_batches, vocab):
        """Compute dynamic vocabs for each example.

        Args:
            input_batches (list[list[list[unicode]]]): a batch of input lists,
                where each input list is a list of sentences
            vocab (HardCopyVocab)

        Returns:
            list[HardCopyDynamicVocab]: a batch of dynamic vocabs, one for each example
        """
        dynamic_vocabs = []
        for input_words in input_batches:
            # compute dynamic vocab from concatenation of input sequences
            #concat = flatten(input_words)
            #dynamic_vocabs.append(HardCopyDynamicVocab(vocab, concat))
            dynamic_vocabs.append(HardCopyDynamicVocab(vocab, input_words, self.copy_lens))
        return dynamic_vocabs

    def forward(self, editor_input):
        """Return the training loss.

        Args:
            editor_input (EditorInput)

        Returns:
            loss (Variable): scalar
            losses (SequenceBatch): of shape (batch_size, seq_length)
        """
        encoder_output, enc_loss = self.encoder(editor_input.encoder_input)
        total_loss, losses = self.train_decoder.loss(encoder_output, editor_input.train_decoder_input)
        return total_loss + enc_loss, losses, enc_loss

    def vocab_probs(self, editor_input):
        """Return raw vocab_probs

        Args:
            editor_input (EditorInput)

        Returns:
            vocab_probs (list[Variable]) contains softmax variables
        """
        encoder_output, enc_loss = self.encoder(editor_input.encoder_input)
        vocab_probs = self.train_decoder.vocab_probs(encoder_output, editor_input.train_decoder_input)
        return vocab_probs

    def loss(self, examples, assert_train=True):
        """Compute loss Variable.

        Args:
            examples (list[EditExample])

        Returns:
            loss (Variable): of shape 1
            loss_traces (list[LossTrace])
        """

        editor_input = self.preprocess(examples)
        total_loss, losses, enc_loss = self(editor_input)
        # list of length batch_size. Each element is a 1D numpy array of per-token losses
        per_ex_losses = list(losses.values.data.cpu().numpy())
        return total_loss, [LossTrace(ex, l, self.base_vocab) for ex, l in
                            zip(examples, per_ex_losses)], enc_loss.data.cpu().numpy()

    def test_batch(self, examples):
        """simple batching test"""
        return
        if len(examples) > 1:
            ex1, ex2 = examples[0:2]

            loss = lambda batch: self.loss(batch, assert_train=False)[0].data.cpu().numpy()

            self.eval()  # test mode, to disable randomness of dropout
            np.random.seed(0)
            torch.manual_seed(0)
            lindivid = loss([ex1]) + loss([ex2])
            np.random.seed(0)
            torch.manual_seed(0)
            ltogether = loss([ex1, ex2]) * 2.0
            self.train()  # set back to train mode

            if abs(lindivid - ltogether)> 1e-3:
                print examples[0:2]
                print 'individually:'
                print lindivid
                print 'batched:'
                print ltogether
                raise Exception('batching error - examples do not produce identical results under batching')
        else:
            raise Exception('test_batch called with example list of length < 2')
        print 'Passed batching test'

    def edit(self, examples, max_seq_length=150, beam_size=5, batch_size=64, constrain_vocab=False, verbose=False):
        """Performs edits on a batch of source sentences.

        Args:
            examples (list[EditExample])
            max_seq_length (int): max # timesteps to generate for
            beam_size (int): for beam decoding
            batch_size (int): max number of examples to pass into the RNN decoder at a time.
                The total # examples decoded in parallel = batch_size / beam_size.
            constrain_vocab (bool):
default is False

        Returns:
            beam_list (list[list[list[unicode]]]): a batch of beams.
            edit_traces (list[EditTrace])
        """
        self.eval()  # set to evaluation mode, for dropout to work correctly
        beam_list = []
        edit_traces = []

        batches = chunks(examples, batch_size / beam_size)
        batches = verboserate(batches, desc='Decoding examples') if verbose else batches
        for batch in batches:
            beams, traces = self._edit_batch(batch, max_seq_length, beam_size, constrain_vocab)
            beam_list.extend(beams)
            edit_traces.extend(traces)
        self.train()  # set back to train mode
        return beam_list, edit_traces

    def _edit_batch(self, examples, max_seq_length, beam_size, constrain_vocab):
        # should only run in evaluation mode
        assert not self.training

        input_words, output_words = self._batch_editor_examples(examples)
        base_vocab = self.base_vocab
        dynamic_vocabs = self._compute_dynamic_vocabs(input_words, base_vocab)
        dynamic_token_embedder = DynamicMultiVocabTokenEmbedder(self.base_source_token_embedder, dynamic_vocabs, base_vocab)

        encoder_input = self.encoder.preprocess(input_words, output_words, dynamic_token_embedder, volatile=True)
        encoder_output, _ = self.encoder(encoder_input)

        extension_probs_modifiers = []

        if constrain_vocab:
            whitelists = [flatten(ex.input_words) for ex in examples]  # will contain duplicates, that's ok
            vocab_constrainer = LexicalWhitelister(whitelists, self.base_vocab, word_to_forms)
            extension_probs_modifiers.append(vocab_constrainer)

        beams, decoder_traces = self.test_decoder_beam.decode(examples, encoder_output,
            beam_size=beam_size, max_seq_length=max_seq_length,
            extension_probs_modifiers=extension_probs_modifiers
        )

        # replace copy tokens in predictions with actual words, modifying beams in-place
        for beam, dyna_vocab in izip(beams, dynamic_vocabs):
            copy_to_word = dyna_vocab.copy_token_to_word
            for i, seq in enumerate(beam):
                beam[i] = [copy_to_word.get(w, w) for w in seq]

        return beams, [EditTrace(ex, d_trace.beam_traces[-1], dyna_vocab)
                       for ex, d_trace, dyna_vocab in izip(examples, decoder_traces, dynamic_vocabs)]

    def interact(self, beam_size=8, constrain_vocab=False, verbose=True):
        ex = EditExample.from_prompt()
        beam_list, edit_traces = self.edit([ex], beam_size=beam_size,
                                           constrain_vocab=constrain_vocab)
        beam = beam_list[0]
        output_words = beam[0]
        edit_trace = edit_traces[0]

        # nll = lambda example: self.loss([example]).data[0]

        # TODO: make this fully generative in the right way.. current NLL is wrong, disabled for now.
        # compare NLL of correct output and predicted output
        # output_ex = EditExample(ex.source_words, ex.insert_words, ex.delete_words, output_words)
        # gold_nll = nll(ex)
        # output_nll = nll(output_ex)

        print 'output:'
        print ' '.join(output_words)

        if verbose:
            # print
            # print 'output NLL: {}, gold NLL: {}'.format(output_nll, gold_nll)
            print edit_trace

    def get_vectors(self, tset):
        """
        :param tset: list of training examples
        :return: vec_list (joint encoding) and vec_list_in (context encoding)
        """
        vec_list = []
        vec_list_in = []
        for titem in chunks(tset, 128):
            edit_proc = self.preprocess(titem, volatile=True)
            agenda_out = self.encoder.target_out(edit_proc.encoder_input)
            agenda_in, _ = self.encoder.ctx_code_out(edit_proc.encoder_input)
            amat = agenda_out.data.cpu().numpy()
            amat_in = agenda_in.data.cpu().numpy()
            for i in range(amat.shape[0]):
                avec = amat[i] + amat_in[i]
                anorm = np.linalg.norm(avec)
                vec_list.append(avec/anorm)
                vec_list_in.append(amat_in[i] / np.linalg.norm(amat_in[i]))
        return vec_list, vec_list_in


