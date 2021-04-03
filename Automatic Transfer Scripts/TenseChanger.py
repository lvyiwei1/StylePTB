from nltk.corpus import treebank
from nltk.tree import Tree
import copy
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
import VerbMorph


def firstvp(tree):
    for i in tree:
        if i.label()[0:2]=='VP':
            return i

def firstword(tree):
    if isinstance(tree,Tree):
        return firstword(tree[0])
    return tree.lower()

def tofuture(tree, parent):
    if isinstance(tree,Tree):
        if firstword(tree) == 'let':
            return
        if tree.label()[0:2]=='VP' and not ((parent.label()[0:2] == 'VP' and tree != parent[0]) or parent.label()[0:3] == 'RRC' or parent.label()[0:5] in ['S-ADV','S-PRD']):
            if tree[0].label()[0:2] == 'TO' or tree[0].label()[0:3]=='VBG' or tree[0].label()[0:3]=='IN':
                for i in range(1,len(tree)):
                    tofuture(tree[i],tree)
            elif tree[0].label() == 'MD':
                if (tree[0][0]=='will' or tree[0][0]=='would'):
                    tree[0]=Tree('MD',['will'])
                elif (tree[0][0]=='could' ):
                    tree[0]=Tree('MD',['can'])
                for i in range(1,len(tree)):
                    tofuture(tree[i],tree)
            elif tree[0].label()== 'VP':
                for i in tree:
                    tofuture(i, tree)
            else:
                tree2 = copy.deepcopy(tree)
                for i in tree2:
                    tree.remove(i)
                tree.insert(0,Tree('MD',['will']))
                #print(tree2)
                tree.insert(1,tree2)

                if tree2[0].label()=='VBD' or tree2[0].label()=='VBZ' or tree2[0].label()=='VB' or tree2[0].label()=='VBP':
                    #print(tree2[0][0])
                    if tree2[0][0]=='fell':
                        tree2[0] = Tree('VB', ['fall'])
                    elif tree2[0][0] in ['\'s','\'S']:
                        tree2[0] = Tree('VB', ['be'])
                    elif tree2[0][0] in ['\'m','\'re']:
                        tree2[0] = Tree('VB', ['be'])
                    else:
                        tree2[0] = Tree('VB',[wnl.lemmatize(tree2[0][0].lower(),'v')])
                    if tree2[0][0] == '\'m' or tree2[0][0] == '\'re' :
                        tree2[0][0] == 'be'
                    if tree2[0][0]=='do':
                        tree2.remove(tree2[0])
                        if len(tree2)==0:
                            tree.remove(tree2)
                        elif tree2[0].label()=='RB':
                            tree.insert(1,tree2[0])
                            tree2.remove(tree2[0])
                    elif tree2[0][0] == 'have':
                        if len(tree2)>2 and tree2[1].label()=='RB':
                            tree.insert(1,tree2[1])
                            tree2.remove(tree2[1])
                    elif tree2[0][0] == 'be' and len(tree2)>1 and tree2[1].label()=='RB':
                        a = tree2[1]
                        tree2.remove(a)
                        tree.insert(1,a)
                    if len(tree)>1 and tree[1][0] == 'n\'t':
                        tree[1][0] = 'not'
                for i in range(1,len(tree2)):
                    tofuture(tree2[i],tree2)
        else:
            for i in tree:
                tofuture(i, tree)

def topast(tree,parent):
    if isinstance(tree,Tree):
        if firstword(tree) == 'let':
            return
        if tree.label()[0:2]=='VP' and not ((parent.label()[0:2] == 'VP' and tree != parent[0]) or parent.label()[0:3] == 'RRC'  or parent.label()[0:5] in ['S-ADV','S-PRD']):
            if tree[0].label()[0:2] == 'TO' or tree[0].label()[0:3]=='VBG':
                for i in range(1,len(tree)):
                    topast(tree[i],tree)
            elif tree[0].label() == 'MD' and (tree[0][0]=='will') :
                if len(tree)>2 and tree[1].label()=='RB' or len(tree)==1 or firstvp(tree)==None:
                    tree[0]=Tree('VBD',['did'])
                    for i in range(1, len(tree)):
                        topast(tree[i], tree)
                else:
                    tree.remove(tree[0])
                    firstv = firstvp(tree)
                    for i in range(0,len(firstv)):
                        tree.insert(i,firstv[i])
                    tree.remove(firstv)
                    topast(tree,parent)
            elif tree[0].label() == 'MD' and (tree[0][0]=='can'):
                tree[0] = Tree('MD',['could'])
            elif tree[0].label()== 'VP':
                for i in tree:
                    topast(i, tree)
            else:
                if tree[0].label()=='VB' or tree[0].label()=='VBZ' or tree[0].label()=='VBN' or tree[0].label()=='VBP':
                    tree[0]=Tree('VBD',[VerbMorph.find_past(halflemmatize(tree[0][0]))])
                    for i in range(1, len(tree)):
                        topast(tree[i], tree)
        else:
            for i in tree:
                topast(i, tree)


def isplural(word):
    if word == 'I' or word == 'we' or word == 'us' or word == 'they' or word == 'them' or word.lower() == "you":
        return True
    lemma = wnl.lemmatize(word, 'n')
    plural = True if word is not lemma else False
    return plural

def issingular(np):
    if np==None:
        return False
    lastnp = None
    for i in np:
        if i.label()[0:2]=='NP':
            lastnp = i
    if lastnp != None:
        return issingular(lastnp)
    nn = None
    for i in np:
        if i.label()[0:2]=='NN' or i.label()=='PRP':
            nn=i[0]
    if nn == None:
        return False
    #print(nn)
    if nn.lower() == 'we':
        return False
    bool = isplural(nn)
    return not bool


def topresent(tree,parent, withnp = None):
    if isinstance(tree,Tree):
        if firstword(tree) == 'let':
            return
        if tree.label()[0:2]=='VP' and not ((parent.label()[0:2] == 'VP' and tree != parent[0]) or parent.label()[0:3] == 'RRC'  or parent.label()[0:5] in ['S-ADV','S-PRD']):
            np = None
            for i in parent:
                if i.label()[0:2]=='NP':
                    np=i
            singular = issingular(np)
            if not withnp == None:
                singular = withnp
            if tree[0].label()[0:2] == 'TO' or tree[0].label()[0:3]=='VBG':
                for i in range(1,len(tree)):
                    topresent(tree[i],tree)
            elif tree[0].label() == 'MD' and (tree[0][0]=='will' or tree[0][0]=='would'):
                if (len(tree)>2 and tree[1].label()=='RB') or len(tree)==1 or firstvp(tree)==None:
                    if singular:
                        tree[0]=Tree('VBZ',['does'])
                    else:
                        tree[0] = Tree('VB', ['do'])
                    for i in range(1, len(tree)):
                        topresent(tree[i], tree)
                else:
                    tree.remove(tree[0])
                    firstv = firstvp(tree)
                    for i in range(0,len(firstv)):
                        tree.insert(i,firstv[i])
                    tree.remove(firstv)
                    topresent(tree,parent)
            elif tree[0].label() == 'MD' and (tree[0][0]=='could'):
                tree[0] = Tree('MD',['can'])
            elif tree[0].label()== 'VP':
                #print(tree[0])
                for i in tree:
                    topresent(i, tree, singular)
            else:
                if tree[0].label() in ['VBD','VBZ','VB','VBP']:
                    if tree[0][0] in ['are','is']:
                        pass
                    elif tree[0][0] == 'be':
                        if singular:
                            tree[0] = Tree('VBZ',['is'])
                        else:
                            tree[0] = Tree('VB',['are'])
                    if singular:
                        tree[0]=Tree('VBZ',[VerbMorph.pluralverb(halflemmatize(tree[0][0]))])
                    else:
                        tree[0] = Tree('VB', [halflemmatize(tree[0][0])])
                    for i in range(1, len(tree)):
                        topresent(tree[i], tree)
        else:
            for i in tree:
                topresent(i, tree)

def halflemmatize(word):
    if word in ['fell','fallen']:
        return 'fall'
    if word in ['is','was']:
        return 'is'
    elif word in ['are','were','\'re']:
        return 'are'
    elif word in ['\'m','am']:
        return 'am'
    elif word in ['\'S','\'s']:
        return 'is'
    out = wnl.lemmatize(word.lower(),'v')
    return out


if __name__ == "__main__":
    count = 0
    for file in treebank.fileids():
        for i in treebank.parsed_sents(file):
            print(i.leaves())
            if i.leaves()[0]=='These' and i.leaves()[1]=='three':
                print(i)
            tofuture(i, Tree('None',[]))
            print(i.leaves())
            count += 1
