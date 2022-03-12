import pickle
from nltk.corpus import treebank
from nltk.tree import Tree
import PassiveActiveVoiceTransfer
import LexicalChange
import TenseChanger
import PPFrontBack
import copy
import re

def sentequal(a,b):
    raw1 = ' '.join(a.leaves()).lower()
    raw2 = ' '.join(b.leaves()).lower()
    return raw1==raw2

def findsame(tokened):
    for file in treebank.fileids():
        for i in treebank.parsed_sents(file):
            if sentequal(tokened, i.leaves()):
                return i
    return None

def getalltrees(filename):
    f = open(filename,'r')
    trees = []
    for i in f.readlines():
        trees.append(Tree.fromstring(i))
    return trees

def treetostring(tree):
    if tree == None:
        return None
    sent = ""
    for word in tree.leaves():
        sent += word + ' '
    return sent

def removeN(string):
    out = ""
    for char in string:
        if not char == '\n':
            out += char
    return out

if __name__ == "__main__":
    trees = getalltrees('ptb-train.txt')
    trees.extend(getalltrees('ptb-test.txt'))
    trees.extend(getalltrees('ptb-valid.txt'))
    transforms = []
    f = open('../../dictionaries/synonym.dict', 'rb')
    syndict = pickle.load(f)
    f = open('../../dictionaries/antonym.dict', 'rb')
    antdict = pickle.load(f)
    count = 0
    for i in trees:
        if i.label()[0] != 'S':
            continue
        if len(i.leaves()) > 12 or len(i.leaves()) < 5:
            continue
        #transforms.append([])
        #print(count)
        count += 1
    print(count)
    """
        print(treetostring(i))
        transforms[-1].append(i)
        j = copy.deepcopy(i)
        PassiveActiveVoiceTransfer.activeToPassive(j)
        if not sentequal(i,j):
            print("activetopassive:"+treetostring(j))
            transforms[-1].append(j)
        else:
            transforms[-1].append(None)
        j = copy.deepcopy(i)
        PassiveActiveVoiceTransfer.passiveToActive(j)
        if not sentequal(i, j):
            print("passivetoactivev:" + treetostring(j))
            transforms[-1].append(j)
        else:
            transforms[-1].append(None)
        j = copy.deepcopy(i)
        TenseChanger.topresent(j, Tree('None',[]))
        if not sentequal(i, j):
            print("topresent:" + treetostring(j))
            transforms[-1].append(j)
        else:
            transforms[-1].append(None)
        j = copy.deepcopy(i)
        TenseChanger.topast(j, Tree('None',[]))
        if not sentequal(i, j):
            print("topast:" + treetostring(j))
            transforms[-1].append(j)
        else:
            transforms[-1].append(None)
        j = copy.deepcopy(i)
        TenseChanger.tofuture(j, Tree('None',[]))
        if not sentequal(i, j):
            print("tofuture:" + treetostring(j))
            transforms[-1].append(j)
        else:
            transforms[-1].append(None)
        j = copy.deepcopy(i)
        j = PPFrontBack.ppfronttoback(j)
        if not sentequal(i, j):
            print("ppfronttoback: " + treetostring(j))
            transforms[-1].append(j)
        else:
            transforms[-1].append(None)
        j = copy.deepcopy(i)
        j = PPFrontBack.ppbacktofront(j)
        if not sentequal(i,j):
            print("ppbacktofront: "+treetostring(j))
            transforms[-1].append(j)
        else:
            transforms[-1].append(None)
        j = copy.deepcopy(i)
        LexicalChange.NounReplacement(j,syndict)
        if not sentequal(i, j):
            print("NounSynonymReplacement: " + treetostring(j))
            transforms[-1].append(j)
        else:
            transforms[-1].append(None)
        j = copy.deepcopy(i)
        LexicalChange.VerbReplacement(j, syndict)
        if not sentequal(i, j):
            print("VerbSynonymReplacement: " + treetostring(j))
            transforms[-1].append(j)
        else:
            transforms[-1].append(None)
        j = copy.deepcopy(i)
        LexicalChange.ADJReplacement(j, syndict)
        if not sentequal(i, j):
            print("ADJSynonymReplacement: " + treetostring(j))
            transforms[-1].append(j)
        else:
            transforms[-1].append(None)
        j = copy.deepcopy(i)
        LexicalChange.NounReplacement(j, antdict)
        if not sentequal(i, j):
            print("NounAntonymReplacement: " + treetostring(j))
            transforms[-1].append(j)
        else:
            transforms[-1].append(None)
        j = copy.deepcopy(i)
        LexicalChange.VerbReplacement(j, antdict)
        if not sentequal(i, j):
            print("VerbAntonymReplacement: " + treetostring(j))
            transforms[-1].append(j)
        else:
            transforms[-1].append(None)
        j = copy.deepcopy(i)
        LexicalChange.ADJReplacement(j, antdict)
        if not sentequal(i, j):
            print("ADJAntonymReplacement: " + treetostring(j))
            transforms[-1].append(j)
        else:
            transforms[-1].append(None)
        print(' ')
    with open('transformeddataset.pk', 'wb+') as f_file:
        pickle.dump(transforms, f_file)
    with open('transformeddataset.txt', 'w+') as f:
        for i in range(0,len(transforms[0])):
            for j in transforms:
                if j[i] == None:
                    f.write("No Change" + '\t')
                else:
                    a = j[i].pformat()
                    a = removeN(a)
                    a = re.sub(' +', ' ', a)
                    #print(a)
                    f.write(a)
                    f.write('\t')
            f.write('\n')
    """




