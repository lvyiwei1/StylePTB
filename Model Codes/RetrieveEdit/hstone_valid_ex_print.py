# -*- coding: utf-8 -*-

import paths
import os
#os.environ['COPY_EDIT_DATA']=paths.data_dir
os.environ['CUDA_VISIBLE_DEVICES']='0'
from gtd.utils import Config, bleu

from editor_code.copy_editor.retrieve_edit_run import RetrieveEditTrainingRun
from editor_code.copy_editor.editor import EditExample
print os.environ['COPY_EDIT_DATA']

import paths
import io

field_delims = ['NAME_END','ATK_END','DEF_END','COST_END','DUR_END','TYPE_END','PLAYER_CLS_END','RACE_END','RARITY_END']
field_prefix = ['','ATK','DEF','COST','DUR','','','','','']

def cut_by_substring(string, field_delims):
    next_start = 0
    subs_list = []
    substring_list = []
    for delim in field_delims:
        delim_start = string.find(delim)
        subs_list.append((next_start, delim_start))
        substring_list.append(string[next_start:delim_start])
        next_start = delim_start + len(delim)
    substring_list.append(string[next_start:(len(string)-1)])
    return substring_list

def load_input(filename):
    lsplit = []
    with io.open(filename+'.in','r') as fopen:
        for line in fopen:
            ssl=cut_by_substring(line.strip(), field_delims)
            slis=[field_prefix[i]+ssl[i].strip()for i in range(len(ssl))]
            lsplit.append(slis)
    return lsplit

def proc_str(strin):
    strin = strin.replace('        ',u'\U0001D7D6')
    strin = strin.replace('    ',u'\U0001D7D2')
    strin = strin.replace('  ',u'\U0001D7D0')
    strin = strin.replace(' ', u'\U0001D7CF')
    return strin

delim_chars = [u'\xa7',u'§',u' ',u'.',u'=',
               u'(',u'\"',u')',u':',u',',u']',u'[',
               u'\U0001D7D6',u'\U0001D7D2',
               u'\U0001D7D0',u'\U0001D7CF']

def tok_str(strin):
    tok=''
    all_list = []
    for i in range(len(strin)):
        if strin[i] in delim_chars:
            if len(tok) > 0:
                all_list.append(tok)
            all_list.append(strin[i])
            tok = ''
        else:
            tok += strin[i]
    return all_list



import regex
def make_eexs(inlist, outlist):
    fline = []
    for instr, outstr in zip(inlist, outlist):
        cardname = regex.sub('[\p{P}\p{Sm}]+', '', ''.join(instr[0].split(' ')))
        i1 = [cardname]+instr[0].split(' ')
        i2 = instr[1:9]
        i3 = instr[9].split(' ')
        tmp=EditExample([i1,i2,i3],outstr)
        fline.append(tmp)
    return fline

import editdistance
def map_vocab(dynamic_vocab, str):
    tmp = []
    for i in range(len(str)):
        if str[i] in dynamic_vocab.copy_token_to_word:
            tmp.append(dynamic_vocab.copy_token_to_word[str[i]])
        else:
            tmp.append(str[i])
    return tmp

def format_ex(exin):
    input = [' '.join(sub) for sub in exin.input_words]
    target = ' '.join(invert_str(exin.target_words))
    ctx_in='\n'.join(['CONTEXT:',input[0],input[1],input[2]])
    ret_in='\n'.join(['RET-CTX:',input[3],input[4],input[5]])
    ret_out='RET-TRG:'+' '.join(invert_str(input[6].split(' ')))
    return '\n'.join([ctx_in, ret_in, ret_out, 'TARGET:'+target])

sub_list = {u'\U0001D7D6':[' ']*8, u'\U0001D7D2':[' ']*4, u'\U0001D7D0':[' ']*2, u'\U0001D7CF':[' ']}
def invert_str(strin):
    tmp = []
    for item in strin:
        if item in sub_list:
            tmp.extend(sub_list[item])
        else:
            tmp.append(item)
    return tmp

import regex as re
def tokenize_for_bleu_eval(code):
    code = re.sub(r'([^A-Za-z0-9_])', r' \1 ', code)
    code = re.sub(r'([a-z])([A-Z])', r'\1 \2', code)
    code = re.sub(r'\s+', ' ', code)
    code = code.replace('"', '`')
    code = code.replace('\'', '`')
    tokens = [t for t in code.split(' ') if t]
    return tokens

def tok_wrapper(strin):
    return tokenize_for_bleu_eval(' '.join(invert_str(strin)))

validation_dir = os.environ['COPY_EDIT_DATA']+'/datasets/card2code/third_party/hearthstone/dev_hs'

output_list = []
with io.open(validation_dir+'.out','r',encoding='utf-8') as fopen:
    for line in fopen:
        output_list.append(line.strip())

out_proc = [tok_str(proc_str(out)) for out in output_list]
iin = load_input(validation_dir)
valid_ex = make_eexs(iin, out_proc)

#no-profile
profile=False

config = Config.from_file('editor_code/configs/editor/default.txt')
src_dir = os.environ['COPY_EDIT_DATA']+'/edit_runs/0'
print 'loading model'
print src_dir
load_expt = RetrieveEditTrainingRun(config,src_dir) #highest valid bleu.

import numpy as np

vae_editor = load_expt.editor.vae_model
ret_model = load_expt.editor.ret_model
edit_model = load_expt.editor.edit_model
examples = load_expt._examples

new_vecs = ret_model.batch_embed(examples.train, train_mode=False)
full_lsh = ret_model.make_lsh(new_vecs)
valid_eval = ret_model.ret_and_make_ex(valid_ex, full_lsh, examples.train, 0, train_mode=False)


beam_list, edit_traces = edit_model.edit(valid_eval,max_seq_length=150,verbose=True, beam_size=5)

edlist = []
gen_out = []
ex_out = []
for i in range(len(edit_traces)):
    trg = edit_traces[i].example.target_words
    gen = beam_list[i][0]
    edlist.append(editdistance.eval(tok_wrapper(gen), tok_wrapper(trg)))
    ex_out.append(edit_traces[i].example)
    gen_out.append(gen)

edsort = np.argsort(edlist)

blist = [bleu(tok_wrapper(edit_traces[i].example.target_words), tok_wrapper(gen_out[i])) for i in range(len(gen_out))]
print 'model BLEU and accuracy'
print np.mean(blist)
print np.mean(np.array(edlist)==0.0)

def print_card(i):
    print str(i)+'------------ new example -------------'+str(edlist[edsort[i]])
    print ' '.join(edit_traces[edsort[i]].example.input_words[0])
    print ' '.join(edit_traces[edsort[i]].example.input_words[1])
    print ' '.join(edit_traces[edsort[i]].example.input_words[2])
    print ' '.join(edit_traces[edsort[i]].example.input_words[3])
    print ' '.join(edit_traces[edsort[i]].example.input_words[4])
    print ' '.join(edit_traces[edsort[i]].example.input_words[5])
    print ' '.join(invert_str(edit_traces[edsort[i]].example.input_words[6]))
    print ' '.join(invert_str(edit_traces[edsort[i]].example.target_words))
    print '------------ generation --------------'
    print ' '.join(invert_str(beam_list[edsort[i]][0]))
    print '\n\n'


for i in range(20):
    print_card(i)

print_card(4)    
