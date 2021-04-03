from collections import namedtuple
from itertools import izip

import numpy as np
import torch
from torch.autograd import Variable

from gtd.ml.torch.utils import GPUVariable, conditional, is_binary, assert_tensor_equal
from gtd.ml.torch.utils import expand_dims_for_broadcast, NamedTupleLike
from gtd.ml.vocab import Vocab


class SequenceBatch(namedtuple('SequenceBatch', ['values', 'mask']), NamedTupleLike):
    """
    Attributes:
        values (Variable): of shape (batch_size, max_seq_length, X1, X2, ...)
        mask (Variable[FloatTensor]): of shape (batch_size, max_seq_length)
    """
    __slots__ = ()
    def __new__(cls, values, mask, left_justify=True):
        if not isinstance(values, Variable) or not isinstance(mask, Variable):
            raise ValueError('values and mask must both be of type Variable.')

        m = mask.data

        if len(m.size()) == 0:
            raise ValueError('Mask must not be 0-dimensional')

        # check that mask is binary
        if not is_binary(m):
            raise ValueError('Mask must be binary:\n{}'.format(mask))

        # check that mask is left-justified
        # since mask is binary, we just need to check that it is monotonically non-increasing from left to right
        batch_size, seq_len = m.size()
        if seq_len > 1 and left_justify:
            diffs = m[:, 1:] - m[:, :-1]  # (batch_size, max_seq_length - 1)
            non_increasing = diffs <= 0
            all_non_increasing = (torch.prod(non_increasing) == 1)
            if not all_non_increasing:
                raise ValueError('Mask must be left-justified:\n{}'.format(mask))

        self = super(SequenceBatch, cls).__new__(cls, values, mask)
        return self

    @classmethod
    def from_sequences(cls, sequences, vocab_or_vocabs, min_seq_length=0, volatile=False):
        """Convert a batch of sequences into a SequenceBatch.

        Args:
            sequences (list[list[unicode]])
            vocab_or_vocabs (WordVocab|list[WordVocab]): either a single vocab, or a list of vocabs, one per sequence
            min_seq_length (int): enforce that the Tensor representing the SequenceBatch have at least
                this many columns.
            volatile (bool): whether to make Variables volatile (don't track grads)

        Returns:
            SequenceBatch
        """
        # determine dimensions
        batch_size = len(sequences)
        if batch_size == 0:
            seq_length = 0
        else:
            seq_length = max(len(seq) for seq in sequences)  # max seq length in batch
        seq_length = max(seq_length, min_seq_length)  # make sure it is at least min_seq_length
        shape = (batch_size, seq_length)

        # set up vocabs
        if isinstance(vocab_or_vocabs, list):
            vocabs = vocab_or_vocabs
            assert len(vocabs) == batch_size
        else:
            # duplicate a single vocab
            assert isinstance(vocab_or_vocabs, Vocab)
            vocabs = [vocab_or_vocabs] * batch_size

        # build arrays
        values = np.zeros(shape, dtype=np.int64)  # pad with zeros
        mask = np.zeros(shape, dtype=np.float32)
        for i, (seq, vocab) in enumerate(izip(sequences, vocabs)):
            for j, word in enumerate(seq):
                values[i, j] = vocab.word2index(word)
                mask[i, j] = 1.0

        return SequenceBatch(GPUVariable(torch.from_numpy(values), volatile=volatile),
                             GPUVariable(torch.from_numpy(mask), volatile=volatile))

    def split(self):
        """Convert SequenceBatch into a list of Variables, where each element represents one time step.

        Returns:
            list[SequenceBatchElement]: a list of SequenceBatchElements, where for each list element:
                element.values has shape (batch_size, X1, X2, ...)
                element.mask has shape (batch_size, 1)
        """
        values_list = [v.squeeze(dim=1) for v in self.values.split(1, dim=1)]
        mask_list = self.mask.split(1, dim=1)
        return [SequenceBatchElement(v, m) for v, m in izip(values_list, mask_list)]

    @classmethod
    def cat(cls, elements):
        """Concatenate SequenceBatchElements to form a SequenceBatch.

        Args:
            elements (list[SequenceBatchElement])

        Returns:
            SequenceBatch
        """
        values = torch.cat([e.values.unsqueeze(1) for e in elements], 1)
        mask = torch.cat([e.mask for e in elements], 1)
        return SequenceBatch(values, mask)

    @classmethod
    def weighted_sum(cls, seq_batch, weights):
        """Compute weighted sum of elements in a SequenceBatch.

        Args:
            seq_batch (SequenceBatch): with values of shape (batch_size, seq_length, X1, X2, ...)
            weights (Variable): of shape (batch_size, seq_length)

        Returns:
            Variable: of shape (batch_size, X1, X2, ...)
        """
        values = seq_batch.values
        mask = seq_batch.mask
        weights = weights * mask  # ignore weights outside mask
        weights = expand_dims_for_broadcast(weights, values).expand(values.size())
        weighted = values * weights
        return torch.sum(weighted, dim=1).squeeze(dim=1)

    @classmethod
    def reduce_sum(cls, seq_batch):
        weights = GPUVariable(torch.ones(*seq_batch.mask.size()))
        return cls.weighted_sum(seq_batch, weights)

    @classmethod
    def reduce_prod(cls, seq_batch):
        """Compute the product of each sequence in a SequenceBatch.
        
        If a sequence is empty, we return a product of 1.
        
        Args:
            seq_batch (SequenceBatch): of shape (batch_size, seq_length, X1, X2, ...)

        Returns:
            Tensor: of shape (batch_size, X1, X2, ...)
        """
        mask = seq_batch.mask
        values = seq_batch.values

        # We set all pad values = 1, so that taking the log will not produce -inf
        mask_bcast = expand_dims_for_broadcast(mask, values).expand(values.size())  # (batch_size, seq_length, X1, X2, ...)
        values = conditional(mask_bcast, values, 1 - mask_bcast)

        logged = SequenceBatch(torch.log(values), seq_batch.mask)  # (batch_size, seq_length, X1, X2, ...)

        log_sum = SequenceBatch.reduce_sum(logged)  # (batch_size, X1, X2, ...)
        prod = torch.exp(log_sum)
        return prod

    @classmethod
    def reduce_mean(cls, seq_batch, allow_empty=False):
        """Compute the mean of each sequence in a SequenceBatch.

        Args:
            seq_batch (SequenceBatch): a SequenceBatch with the following attributes:
                values (Tensor): a Tensor of shape (batch_size, seq_length, X1, X2, ...)
                mask (Tensor): if the mask values are arbitrary floats (rather than binary), the mean will be
                a weighted average.
            allow_empty (bool): allow computing the average of an empty sequence. In this case, we assume 0/0 == 0, rather
                than NaN. Default is False, causing an error to be thrown.

        Returns:
            Tensor: of shape (batch_size, X1, X2, ...)
        """
        values, mask = seq_batch.values, seq_batch.mask
        # compute weights for the average
        sums = torch.sum(mask, dim=1)  # (batch_size, 1)

        if allow_empty:
            sums[sums == 0.0] = 1.0  # Modify in-place: replace zeros with ones
        else:
            if (sums.data == 0).any():
                raise ValueError("Averaging zero elements.")

        weights = mask / sums.expand(*mask.size())
        return cls.weighted_sum(seq_batch, weights)

    @classmethod
    def _empty_seqs(cls, seq_batch):
        return (torch.sum(seq_batch.mask, 1).data == 0).any()

    @classmethod
    def reduce_max(cls, seq_batch):
        if cls._empty_seqs(seq_batch):
            raise ValueError("Taking max over zero elements.")
        values, mask = seq_batch.values, seq_batch.mask

        inf_mask = mask.clone()  # (batch_size, seq_length)
        inf_mask[mask == 0] = float('inf')
        inf_mask[mask == 1] = 0
        # masked elements will never win the max, because we subtract infinity from them

        inf_mask_bcast = expand_dims_for_broadcast(inf_mask, values).expand_as(values)  # (batch_size, seq_length, X1, X2, ...)

        max_values, _ = torch.max(values - inf_mask_bcast, 1)  # (batch_size, 1, X1, X2, ...)
        max_values = torch.squeeze(max_values, 1)  # (batch_size, X1, X2, ...)

        return max_values

    @classmethod
    def log_sum_exp(cls, seq_batch):
        """Numerically stable computation of log-sum-exp.
        
        Mask must be left-justified.
        Does not allow empty rows.
        
        x = seq_batch.values
        lse[i] = log(exp(x[i, 0]) + exp(x[i, 1]) + ... + exp(x[i, n]))
        
        Uses the log-sum-exp stability trick:
        https://hips.seas.harvard.edu/blog/2013/01/09/computing-log-sum-exp/
        
        Args:
            seq_batch (SequenceBatch): of shape (batch_size, seq_length), where seq_batch.values of type FloatTensor

        Returns:
            lse (Variable): of shape (batch_size,)
        """
        values, mask = seq_batch.values, seq_batch.mask
        max_vals = cls.reduce_max(seq_batch)  # (batch_size,)
        shifted_values = values - max_vals.unsqueeze(1).expand(values.size())  # (batch_size, seq_length)
        exponentiated_values = torch.exp(shifted_values)
        sums = cls.reduce_sum(SequenceBatch(exponentiated_values, mask))  # (batch_size)
        log_sums = torch.log(sums)
        return log_sums + max_vals

    @classmethod
    def embed(cls, indices, embeds):
        """Embed a SequenceBatch of integers.
        
        Args:
            indices (SequenceBatch): of shape (batch_size, seq_length), with seq_batch.values of type LongTensor (ints)
            embeds (Variable): of shape (vocab_size, embed_dim)

        Returns:
            SequenceBatch: of shape (batch_size, seq_length, embed_dim)
        """
        values, mask = indices

        batch_size, seq_length = values.size()
        vocab_size, embed_dim = embeds.size()

        indices_flat = values.view(batch_size * seq_length)
        embedded_indices_flat = torch.index_select(embeds, 0, indices_flat)  # (batch_size * seq_length, embed_dim)
        embedded_indices = embedded_indices_flat.view(batch_size, seq_length, embed_dim)
        return SequenceBatch(embedded_indices, mask)

    @classmethod
    def multi_vocab_indices(self, sequences, vocabs, min_seq_length=0):
        """Convert a batch of sequences into indices, where each token is converted into a **tuple** of indices.

        Args:
            sequences (list[list[unicode]]): a batch of sequences
            vocabs (list[list[Vocab]]): vocabs[v] = a batch of vocabs (one per example in the batch), corresponding to vocab v
            min_seq_length (int): see SequenceBatch.from_sequences
        
        Returns:
            SequenceBatch:
                mask (Variable): has shape (batch_size, seq_length)
                values (Variable): has shape (batch_size, seq_length, num_vocabs)
        """
        seq_batches = [SequenceBatch.from_sequences(sequences, vocabs_v, min_seq_length) for vocabs_v in vocabs]
        values_list, mask_list = zip(*seq_batches)

        for mask in mask_list:
            assert_tensor_equal(mask, mask_list[0])  # all masks should be the same

        values = torch.stack(values_list, 2)

        return SequenceBatch(values, mask_list[0])


SequenceBatchElement = namedtuple('SequenceBatchElement', ['values', 'mask'])
