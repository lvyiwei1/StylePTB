from nltk.corpus import treebank
from nltk.tree import Tree
import PTBdata

def ADJRBRemoval(tree):
    jjs = []
    if not (len(tree.label())>=4 and tree.label()[0:4] in ['ADJP','ADVP']) and not(len(tree)==1 and tree.label()[0:2]=='NP'):
        for i in tree:
            if isinstance(i,Tree) and i.label()[0:2]=='JJ':
                jjs.append(i)
        for jj in jjs:
            tree.remove(jj)
    jjs = []
    for i in tree:
        if isinstance(i, Tree) and i.label()[0:2] == 'RB' and i[0] not in ['not','n\'t']:
            jjs.append(i)
    for jj in jjs:
        tree.remove(jj)
    jjs = []
    for i in tree:
        if isinstance(i,Tree):
            ADJRBRemoval(i)
            if len(i) == 0:
                jjs.append(i)
    for jj in jjs:
        tree.remove(jj)

def SBARRemoval(tree):
    rms = []
    for i in tree:
        if isinstance(i,Tree) and len(i.label()) >= 4 and i.label()[0:4]=='SBAR':
            rms.append(i)
        elif isinstance(i,Tree) and len(i.label()) >= 5 and i.label()[0:5]=='S-ADV':
            rms.append(i)
    for rm in rms:
        tree.remove(rm)
    for i in tree:
        if isinstance(i,Tree):
            SBARRemoval(i)

def PPRemoval(tree):
    rms = []
    for i in tree:
        if isinstance(i, Tree) and i.label()[0:2] == 'PP':
            rms.append(i)
    for rm in rms:
        tree.remove(rm)
    for i in tree:
        if isinstance(i, Tree):
            PPRemoval(i)

#def SubstatementRemoval(tree):




if __name__ == "__main__":
    trees = PTBdata.getalltrees('ptb-train.txt')
    trees.extend(PTBdata.getalltrees('ptb-test.txt'))
    trees.extend(PTBdata.getalltrees('ptb-valid.txt'))
    for i in trees:
        if len(i.leaves())>20:
            continue
        if i.label()[0] != 'S':
            continue
        print(' '.join(i.leaves()))
        SBARRemoval(i)
        print(' '.join(i.leaves()))

