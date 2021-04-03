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

#Note - we need this or else the program crashes due to a utf-8 error when trying to pipe the outputs to a text file.
#set_output_encoding()

from gtd.utils import Config

from editor_code.copy_editor.edit_training_run import EditTrainingRun
from editor_code.copy_editor.retrieve_edit_run import RetrieveEditTrainingRun
from editor_code.copy_editor.editor import EditExample
from editor_code.copy_editor.vocab import HardCopyDynamicVocab

from gtd.utils import bleu

print os.environ['COPY_EDIT_DATA']

# no-profile
profile = False

config = Config.from_file('editor_code/configs/editor/github.txt')

src_dir_noret = os.environ['COPY_EDIT_DATA']+'/edit_runs/1' #for codalab
load_expt_noret = EditTrainingRun(config, src_dir_noret)
src_dir = os.environ['COPY_EDIT_DATA']+'/edit_runs/0' #for codalab
load_expt = RetrieveEditTrainingRun(config, src_dir)

###
# retedit model
import numpy as np

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
valid_eval = ret_model.ret_and_make_ex(examples.valid[0:eval_num], new_lsh, examples.train, 0, train_mode=False)
beam_list, edit_traces = edit_model.edit(valid_eval)

### other
edit_model_noret = load_expt_noret.editor
beam_list_noret, edit_traces_noret = edit_model_noret.edit(examples.valid[0:eval_num])

###
# base retriever.
import gtd.retrieval_func as rf
lsh, dict = rf.make_hash(examples.train)
output_index = rf.grab_nbs(examples.valid[0:eval_num], lsh, dict)
ret_pred = rf.generate_predictions(examples.train, output_index)

####
# eval code
gen_out = []
gen2_out = []
ret_trout = []
ret_fix_out = []
for i in range(len(edit_traces_noret)):
    ret_fix_out.append(ret_pred[i])
    ret_trout.append(edit_traces[i].example.input_words[6])
    gen = beam_list[i][0]
    gen2 = beam_list_noret[i][0]
    gen_out.append(gen)
    gen2_out.append(gen2)

print examples.valid[386]
print ' '.join(gen_out[386])
print ' '.join(gen2_out[386])
print ' '.join(ret_trout[386])
print ' '.join(ret_fix_out[386])
print '\n'

