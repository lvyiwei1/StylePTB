# -*- coding: utf-8 -*-

import paths
import os

#os.environ['COPY_EDIT_DATA'] = paths.data_dir
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


import sys
def set_output_encoding(encoding='utf-8'):
    import sys
    import codecs
    '''When piping to the terminal, python knows the encoding needed, and
       sets it automatically. But when piping to another program (for example,
       | less), python can not check the output encoding. In that case, it 
       is None. What I am doing here is to catch this situation for both 
       stdout and stderr and force the encoding'''
    current = sys.stdout.encoding
    if current is None :
        sys.stdout = codecs.getwriter(encoding)(sys.stdout)
    current = sys.stderr.encoding
    if current is None :
        sys.stderr = codecs.getwriter(encoding)(sys.stderr)

#Note - we need this or else the program crashes due to a utf-8 error when trying to pipe the outputs to a text file. comment this line out when running interactively
set_output_encoding()

from gtd.utils import Config

from editor_code.copy_editor.retrieve_edit_run import RetrieveEditTrainingRun
from editor_code.copy_editor.editor import EditExample
from editor_code.copy_editor.vocab import HardCopyDynamicVocab

from gtd.utils import bleu

print os.environ['COPY_EDIT_DATA']

# no-profile
profile = False

config = Config.from_file('editor_code/configs/editor/github.txt')
src_dir = os.environ['COPY_EDIT_DATA']+'/edit_runs/0' #for codalab
load_expt = RetrieveEditTrainingRun(config, src_dir)

import numpy as np

vae_editor = load_expt.editor.vae_model
ret_model = load_expt.editor.ret_model
edit_model = load_expt.editor.edit_model
examples = load_expt._examples

from gtd.utils import chunks
from tqdm import tqdm

new_vecs = []
for batch in tqdm(chunks(examples.train,32), total=len(examples.train)/32):
    encin = ret_model.encode(batch, train_mode=False).data.cpu().numpy()
    for vec in encin:
        new_vecs.append(vec)
    del encin

new_lsh = ret_model.make_lsh(new_vecs)

eval_num = 500
valid_eval = ret_model.ret_and_make_ex(examples.test[0:eval_num], new_lsh, examples.train, 0, train_mode=False)
beam_list, edit_traces = edit_model.edit(valid_eval)

import numpy as np
def eval_batch_noret(ex):
    edit_model_noret.copy_index=0
    editor_input = edit_model_noret.preprocess(ex)
    train_decoder = edit_model_noret.train_decoder
    encoder_output, enc_loss = edit_model_noret.encoder(editor_input.encoder_input)
    vocab_probs = edit_model_noret.train_decoder.vocab_probs(encoder_output, editor_input.train_decoder_input)
    token_list = editor_input.train_decoder_input.target_words.split()
    base_vocab = edit_model_noret.base_vocab
    unk_idx = base_vocab.word2index(base_vocab.UNK)
    all_ranks_noret = [ [] for _ in range(len(ex))]
    position = 0
    for token, vout in zip(token_list, vocab_probs):
        target_idx = token.values.data.cpu().numpy()
        target_mask = token.mask.data.cpu().numpy()
        in_vocab_id = target_idx[:,0]
        copy_token_id = target_idx[:, 1]
        vocab_matrix = vout.data.cpu().numpy()
        for i in range(len(in_vocab_id)):
            voc_vec = vocab_matrix[i,:].copy()
            voc_vec_rest = voc_vec.copy()
            voc_vec_rest[copy_token_id[i]]=0
            voc_vec_rest[in_vocab_id[i]] = 0
            if in_vocab_id[i] == unk_idx:
                gold_rank = np.sum(voc_vec_rest >= voc_vec[copy_token_id[i]])
            else:
                gold_rank = np.sum(voc_vec_rest >= voc_vec[copy_token_id[i]] + voc_vec[in_vocab_id[i]])
            if target_mask[i] == 1.0:
                all_ranks_noret[i].append(gold_rank)
        position+=1
    del token_list
    del vocab_probs
    return all_ranks_noret


def eval_batch_ret(ex):
    editor_input = edit_model.preprocess(ex)
    train_decoder = edit_model.train_decoder
    encoder_output, enc_loss = edit_model.encoder(editor_input.encoder_input)
    vocab_probs = edit_model.train_decoder.vocab_probs(encoder_output, editor_input.train_decoder_input)
    token_list = editor_input.train_decoder_input.target_words.split()
    base_vocab = edit_model.base_vocab
    unk_idx = base_vocab.word2index(base_vocab.UNK)
    idx_lists = []
    for k in range(len(ex)):
        hcdv = HardCopyDynamicVocab(base_vocab, valid_eval[k].input_words, edit_model.copy_lens)
        copy_tok_list = [hcdv.word_to_copy_token.get(tok,base_vocab.UNK) for tok in valid_eval[k].input_words[6]]
        copy_tok_id = [hcdv.word2index(tok) for tok in copy_tok_list]
        idx_lists.append(copy_tok_id)
    ret_mix_pr = 0.0
    all_ranks = [ [] for _ in range(len(ex))]
    all_ranks_ret =[ [] for _ in range(len(ex))]
    position = 0
    for token, vout in zip(token_list, vocab_probs):
        target_idx = token.values.data.cpu().numpy()
        target_mask = token.mask.data.cpu().numpy()
        in_vocab_id = target_idx[:,0]
        copy_token_id = target_idx[:, 1]
        vocab_matrix = vout.data.cpu().numpy()
        for i in range(len(in_vocab_id)):
            voc_vec = vocab_matrix[i,:].copy()
            voc_vec_rest = voc_vec.copy()
            voc_vec_rest[copy_token_id[i]] = 0
            voc_vec_rest[in_vocab_id[i]] = 0
            if position < len(idx_lists[i]):
                direct_copy_idx = idx_lists[i][position]
                voc_vec = voc_vec*(1-ret_mix_pr)
                voc_vec[direct_copy_idx] += ret_mix_pr
            if in_vocab_id[i] == unk_idx:
                gold_rank = np.sum(voc_vec_rest >= voc_vec[copy_token_id[i]])
            else:
                gold_rank = np.sum(voc_vec_rest >= voc_vec[copy_token_id[i]] + voc_vec[in_vocab_id[i]])
            if target_mask[i] == 1.0:
                all_ranks[i].append(gold_rank)
                all_ranks_ret[i].append(100*(1.0-(direct_copy_idx == copy_token_id[i])))
        position+=1
    del token_list
    del vocab_probs
    return all_ranks, all_ranks_ret

all_ranks = []
for chunk in tqdm(chunks(valid_eval[0:eval_num],16), total=eval_num/16):
    all_ranks.extend(eval_batch_ret(chunk)[0])

###
# base retriever.
import gtd.retrieval_func as rf
lsh, dict = rf.make_hash(examples.train)
output_index = rf.grab_nbs(examples.test[0:eval_num], lsh, dict)
ret_pred = rf.generate_predictions(examples.train, output_index)

def agree_vec(ref, targ):
    rank_vec = []
    for i in range(max(len(ref),len(targ))):
        if i < len(targ) and i < len(ref):
            agree_ind = ref[i] == targ[i]
            rank_vec.append((1.0-agree_ind)*100.0)
        else:
            rank_vec.append(100.0)
    return rank_vec

all_ranks_ret_fixed = []
for i in range(eval_num):
    all_ranks_ret_fixed.append(agree_vec(examples.test[i].target_words, ret_pred[i]))

all_ranks_ret = []
for i in range(eval_num):
    all_ranks_ret.append(agree_vec(examples.test[i].target_words, valid_eval[i].input_words[6]))

####
# eval code

import itertools

def rle(inarray):
    """ run length encoding. Partial credit to R rle function.
        Multi datatype arrays catered for including non Numpy
        returns: tuple (runlengths, startpositions, values) """
    ia = np.asarray(inarray)  # force numpy
    n = len(ia)
    if n == 0:
        return (None, None, None)
    else:
        y = np.array(ia[1:] != ia[:-1])  # pairwise unequal (string safe)
        i = np.append(np.where(y), n - 1)  # must include last element posi
        z = np.diff(np.append(-1, i))  # run lengths
        p = np.cumsum(np.append(0, z))[:-1]  # positions
        return (z, p, ia[i])


def avg_runlen(rankin, cut):
    tmp = []
    for rank in rankin:
        if sum(np.array(rank) <= cut) > 0:
            rlevals = rle(np.array(rank) <= cut)
            match_pr = rlevals[0][rlevals[2]] / float(np.sum(rlevals[0])) #probability of picking each run
            expect_dist = (rlevals[0][rlevals[2]]+1.0)/2.0 #expected run length over each run (if we sample uniformly)
            elen = np.sum(np.array(expect_dist)*np.array(match_pr))
            tmp.append(elen)
        else:
            tmp.append(0)
    return tmp

def correct_runlen(rankin, cut):
    tmp = []
    for rank in rankin:
        rlevals = rle(np.array(rank) <= cut)
        if np.sum(rlevals[2])>0:
            tmp.append(np.max(rlevals[0][rlevals[2]]))
        #if rlevals[2][0]:
        #    tmp.append(rlevals[0][0])
        else:
            tmp.append(0)
    return tmp


eval_fns = [lambda x: np.mean(avg_runlen(x, 1)), lambda x: np.mean(avg_runlen(x, 5)), lambda x: np.mean(avg_runlen(x, 10)),
            lambda x: np.mean(correct_runlen(x, 1)), lambda x: np.mean(correct_runlen(x, 5)), lambda x: np.mean(correct_runlen(x, 10))]

methods = [all_ranks, all_ranks_ret, all_ranks_ret_fixed]

all_eval = [[fn(method) for method in methods] for fn in eval_fns]
print 'method table: ret+edit, ret_only, ret_fixed'
print np.array(all_eval)

import regex as re
def tokenize_for_bleu_eval(code):
    code = re.sub(r'([^A-Za-z0-9_])', r' \1 ', code)
    code = re.sub(r'([a-z])([A-Z])', r'\1 \2', code)
    code = re.sub(r'\s+', ' ', code)
    code = code.replace('"', '`')
    code = code.replace('\'', '`')
    tokens = [t for t in code.split(' ') if t]
    return tokens

import editdistance
edlist = []
gen_out = []
gen2_out = []
ex_out = []
ret_trout = []
ret_fix_out = []
ret_vae_out = []
for i in range(len(edit_traces)):
    trg = edit_traces[i].example.target_words
    ret_trout.append(edit_traces[i].example.input_words[6])
    ret_fix_out.append(ret_pred[i])
    gen = beam_list[i][0]
    edlist.append(editdistance.eval(gen, trg))
    ex_out.append(edit_traces[i].example)
    gen_out.append(gen)

def btok(x,y):
    return bleu(tokenize_for_bleu_eval(' '.join(x)),tokenize_for_bleu_eval(' '.join(y)))

print 'BLEU'
orig_out = [trace.example.target_words for trace in edit_traces]
blist_gen = [btok(gen_out[i],orig_out[i]) for i in range(len(gen_out))]
print 'ret+edit'
print np.mean(blist_gen)

blist_ret = [btok(ret_trout[i], orig_out[i]) for i in range(len(gen_out))]
print 'trained ret'
print np.mean(blist_ret)

blist_ret_fix = [btok(ret_fix_out[i], orig_out[i]) for i in range(len(gen_out))]
print 'fixed ret'
print np.mean(blist_ret_fix)

