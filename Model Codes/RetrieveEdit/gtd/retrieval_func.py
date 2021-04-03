from collections import defaultdict
from annoy import AnnoyIndex
from tqdm import tqdm
import numpy as np

def map_sent(sent, proj_mat, ndim):
    ssub = sent  # .split(' ')
    tempvec = np.zeros(ndim)
    for word in ssub:
        tempvec += proj_mat[word]
    return tempvec


def make_hash(examples_list):
    np.random.seed(0)
    ndim = 256
    proj_mat = defaultdict(lambda: np.random.randn(ndim))
    t = AnnoyIndex(ndim, metric='angular')
    lnum = 0
    for ex in tqdm(examples_list):
        sent = ex.input_words[1]
        sv = map_sent(sent, proj_mat, ndim)
        t.add_item(lnum, sv)
        lnum += 1
    t.build(10)
    return t, proj_mat

def make_hash_from_vec(vecs):
    np.random.seed(0)
    t = AnnoyIndex(len(vecs[0]), metric='angular')
    for lnum, vec in tqdm(enumerate(vecs)):
        t.add_item(lnum, vec)
    t.build(10)
    return t

def grab_nbs(test_ex_list, lsh, proj_mat, ndim=256):
    nbs_list = []
    for ex in tqdm(test_ex_list):
        sv_test = map_sent(ex.input_words[1], proj_mat, ndim)
        nbs_list.append(lsh.get_nns_by_vector(sv_test, 1)[0])
    return nbs_list

def grab_nbs_from_vec(vecin, lsh):
    nbs_list = []
    for vec in tqdm(vecin):
        nbs_list.append(lsh.get_nns_by_vector(vec, 1)[0])
    return nbs_list


def generate_predictions(train_list, idx):
    return [train_list[idx[i]].target_words for i in range(len(idx))]

