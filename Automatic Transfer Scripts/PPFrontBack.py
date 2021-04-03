from nltk.corpus import treebank
from nltk.tree import Tree

def lastnonpunct(tree):
    last = 0
    for i in range(0,len(tree)):
        if tree[i].label() not in [',','?','.','!',':']:
            last = i
    return last

def lastvp(tree):
    last = 0
    for i in range(0,len(tree)):
        if tree[i].label()[0:2]=='VP':
            last = i
    return last


def ppfronttoback(tree):
    if hasppfront(tree):
        if len(tree)>2 and tree[2].label()[0]=='S' and tree[1].label() == ',':
            lastv = lastvp(tree[2])
            tree[2][lastv].insert(lastnonpunct(tree[2][lastv])+1, tree[0])
        else:
            last = lastnonpunct(tree)
            tree[last].insert(len(tree[last]), tree[0])
        tree.remove(tree[0])
        if len(tree)==0:
            return None
        if tree[0].label()==',':
            tree.remove(tree[0])
        return tree
    else:
        return None

def ppbacktofront(tree):
    backpp = hasppback(tree,3)
    last = lastnonpunct(tree)
    if  not backpp == None:

        if tree[last].label()[0:2]=='PP' or tree[last].label()=='S-ADV':
            tree.remove(backpp)
            tree.insert(0,backpp)
            if tree[last].label()==',':
                tree.pop(last)
                tree.insert(1,Tree(',',[',']))
        else:
            tree[last].remove(backpp)
            tree.insert(0,backpp)
            tree.insert(1,Tree(',',[',']))
    else:
        return None
    return tree

def hasppfront(tree):
    return tree[0].label()[0:2] == 'PP' or tree[0].label() == 'S-ADV'

def hasppback(tree, depth):
    d = 0
    if len(tree) <= 2:
        return None
    t = tree
    while len(t)>0 and d < depth:
        d+= 1
        if isinstance(t[-1],Tree):
            t = t[lastnonpunct(t)]
        if(t.label()[0:2]=='PP' or t.label() == 'S-ADV'):
            return t
        else:
            break
    return None



if __name__ == "__main__":
    for file in treebank.fileids():
        for i in treebank.parsed_sents(file):
            if not i.label()[0]=='S':
                continue
            if not hasppback(i,2) == None:
                print(ppbacktofront(i))