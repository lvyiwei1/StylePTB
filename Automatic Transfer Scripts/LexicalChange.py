from nltk.corpus import treebank
from nltk.tree import Tree
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
from nltk.corpus import wordnet as wn
import VerbMorph
import PTBdata

import pickle

#selects first synonym
def defaultselectfunc(l):
    if len(l)>1:
        return l[1]
    return l[0]

def checkpos(word,pos):
    syn = wn.synsets(word)
    if len(syn) == 0:
        return False
    tmp = syn[0].pos()
    return tmp == pos

def ADJReplacement(tree,dict,selectfunc=defaultselectfunc, limit = 3):
    if limit == 0:
        return 0
    if isinstance(tree,Tree):
        for i in tree:
            if isinstance(i,Tree) and i.label() in ['JJ']:
                lemma = wnl.lemmatize(i[0])
                replace = lemma
                if lemma in dict and len(dict[lemma])>0:
                    replace = selectfunc(dict[lemma])
                if checkpos(replace,'a') and not i[0]==replace:
                    limit -= 1
                    i[0]=replace
            else:
                limit = ADJReplacement(i,dict,selectfunc,limit)
        return limit
    else:
        return limit

def VerbReplacement(tree,dict,selectfunc=defaultselectfunc, limit = 3):
    if limit == 0:
        return 0
    if isinstance(tree,Tree):
        for i in tree:
            if isinstance(i,Tree) and i.label() in ['VB','VBZ','VBD','VBN'] and i[0] not in ['have','has','had']:
                lemma = wnl.lemmatize(i[0])
                replace = lemma
                if lemma in dict and not lemma == 'be' and len(dict[lemma])>0:
                    replace = selectfunc(dict[lemma])
                if checkpos(replace,'v') and not i[0]==replace:
                    if i.label() == 'VBZ':
                        replace = VerbMorph.pluralverb(replace)
                    elif i.label() == 'VBD':
                        replace = VerbMorph.find_past(replace)
                    elif i.label() == 'VBN':
                        replace = VerbMorph.find_past_participle(replace)
                    limit -= 1
                    i[0]=replace
            else:
                limit = VerbReplacement(i,dict,selectfunc,limit)
        return limit
    else:
        return limit

def NounReplacement(tree,dict,selectfunc=defaultselectfunc, limit = 3):
    if limit == 0:
        return 0
    if isinstance(tree,Tree):
        for i in tree:
            if isinstance(i,Tree) and i.label() in ['NN','NNS']:
                lemma = wnl.lemmatize(i[0])
                replace = lemma
                if lemma in dict and len(dict[lemma])>0:
                    replace = selectfunc(dict[lemma])
                if checkpos(replace,'n') and not i[0]==replace:
                    if not lemma == i[0]:
                        replace = VerbMorph.pluralverb(replace)
                    limit -= 1
                    i[0]=replace
            else:
                limit = NounReplacement(i,dict,selectfunc,limit)
        return limit
    else:
        return limit


def makeFrequency(trees):
    freqdict = {}
    for tree in trees:
        for word in tree.leaves():
            word = word.lower()
            if word not in freqdict:
                freqdict[word]=0
            freqdict[word]+=1
    return freqdict

def freqdictselector(words,freqdict,freqlvl):
    for word in words:
        if word not in freqdict:
            freqdict[word]=0
    sorter = lambda x : freqdict[x]
    words.sort(key=sorter)
    place = round(float(len(words)-1)/4.0*float(freqlvl-1),0)
    return words[int(place)]




def FrequencySynonymReplacement(tree,dict,freqdict,freqlvl):
    if isinstance(tree,Tree):
        for i in tree:
            if isinstance(i,Tree) and i.label() in ['NN','NNS']:
                lemma = wnl.lemmatize(i[0].lower())
                replace = lemma
                if lemma in dict and len(dict[lemma])>0:
                    replaces = dict[lemma]
                    replace = freqdictselector(replaces,freqdict,freqlvl)
                    if checkpos(replace,'n') and not i[0]==replace:
                        if not lemma == i[0]:
                            replace = VerbMorph.pluralverb(replace)
                        i[0]=replace
            elif isinstance(i,Tree) and i.label() in ['VB','VBZ','VBD','VBN'] and i[0] not in ['have','has','had']:
                lemma = wnl.lemmatize(i[0].lower())
                replace = lemma
                if lemma in dict and not lemma == 'be' and len(dict[lemma])>0:
                    replaces = dict[lemma]
                    replace = freqdictselector(replaces, freqdict, freqlvl)
                    if checkpos(replace,'v') and not i[0]==replace:
                        if i.label() == 'VBZ':
                            replace = VerbMorph.pluralverb(replace)
                        elif i.label() == 'VBD':
                            replace = VerbMorph.find_past(replace)
                        elif i.label() == 'VBN':
                            replace = VerbMorph.find_past_participle(replace)
                        i[0]=replace
            elif isinstance(i,Tree) and i.label() in ['JJ']:
                lemma = wnl.lemmatize(i[0].lower())
                replace = lemma
                if lemma in dict and len(dict[lemma])>0:
                    replaces = dict[lemma]
                    replace = freqdictselector(replaces, freqdict, freqlvl)
                    if checkpos(replace,'a') and not i[0]==replace:
                        i[0]=replace
            else:
                FrequencySynonymReplacement(i,dict,freqdict,freqlvl)






import create2koriginal
import copy
if __name__ == "__main__":
    f = open('../../dictionaries/synonym.dict','rb')
    dict = pickle.load(f)
    """
    for file in treebank.fileids():
        for i in treebank.parsed_sents(file):
            print(i)
            ADJReplacement(i,dict)
            print(i)
            count += 1
            if count == 5:
                break
        if count == 5:
            break
    """
    trees = PTBdata.getalltrees('ptb-train.txt')
    trees.extend(PTBdata.getalltrees('ptb-test.txt'))
    trees.extend(PTBdata.getalltrees('ptb-valid.txt'))
    freqdict=makeFrequency(trees)
    print(freqdictselector(dict['angry'],freqdict,1))
    count=0
    for tree in trees:
        if len(tree.leaves()) < 5 or len(tree.leaves()) > 12 or not tree.label()[0] == 'S':
            continue
        j = copy.deepcopy(tree)
        FrequencySynonymReplacement(j,dict,freqdict,1)
        if  tree.leaves()==j.leaves():
            continue
        count += 1
        if count < 30:
            pass
            #print(tree.leaves())
            #print(j.leaves())
    print(count)







