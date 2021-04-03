from nltk.corpus import treebank
from nltk.tree import Tree
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
import VerbMorph
import TenseChanger

def isbe(word):
    return word in ['is','are','be','was','were','been']

def findfirst(tree,str):
    for i in tree:
        if i.label()[0:len(str)]==str and not i.label() == 'NP-TMP':
            return i
    return None


def tosub(tree, outer = True):
    if outer:
        tree.label()=='NP-SBJ'
    for i in tree:
        if i.label()[0:2]=='NP':
            tosub(i,False)
        elif i.label()[0:3] == 'PRP':
            if i[0].lower()  == 'us':
                i[0] = 'we'
            if i[0].lower()  == 'me':
                i[0] = 'I'
            if i[0].lower()  == 'him':
                i[0] = 'he'
            if i[0].lower()  == 'her':
                i[0] = 'she'
            if i[0].lower()  == 'them':
                i[0] = 'they'
    return tree

def toobj(tree, outer = True):
    if outer and tree.label()[0:6]=='NP-SBJ':
        tree.set_label('NP')
    for i in tree:
        if i.label()[0:2]=='NP':
            toobj(i,False)
        elif i.label()[0:3] == 'PRP':
            #print(i[0])
            if i[0].lower()  == 'we':
                i[0] = 'us'
            if i[0] == 'I':
                i[0] = 'me'
            if i[0].lower()  == 'he':
                i[0] = 'him'
            if i[0].lower()  == 'she':
                i[0] = 'her'
            if i[0].lower() == 'they':
                i[0] = 'them'
    return tree

def findall(tree,str):
    ret = []
    for i in tree:
        if i.label()[0:len(str)]==str:
            ret.append(i)
    return ret

def passiveToActive(tree):
    if isinstance(tree,Tree):
        if TenseChanger.firstword(tree) == 'let':
            return
        if tree.label()[0]=='S'  and not tree.label() == 'SYM':
            firstnp = -1
            firstvp = -1
            for i in range(0,len(tree)):
                if tree[i].label()[0:2]=='NP' and firstnp < 0:
                    firstnp=i
                if tree[i].label()[0:2] == 'VP' and firstvp < 0:
                    firstvp = i
            if firstnp >= 0 and firstvp >= 0:
                vp1 = tree[firstvp]
                firstvb = findfirst(vp1,'VB')
                if firstvb == None:
                    return
                if isbe(firstvb[0]):
                    nextvp = findfirst(vp1,'VP')
                    if nextvp == None:
                        for i in tree:
                            passiveToActive(i)
                    else:
                        vbn = findfirst(nextvp,'VBN')
                        if vbn==None:
                            for i in tree:
                                passiveToActive(i)
                        else:
                            vbnpos = nextvp.index(vbn)
                            pps = findall(nextvp, 'PP')
                            hasby = False
                            for pp in pps:
                                if pp[0][0]=='by':
                                    obj = findfirst(pp,'NP')
                                    if not obj == None:
                                        objpos = pp.index(obj)
                                        hasby=True
                                        break
                            if not hasby:
                                for i in tree:
                                    passiveToActive(i)
                            else:
                                sub = tree[firstnp]
                                tree.remove(sub)
                                tree.insert(firstnp, tosub(obj))
                                if findfirst(vp1,'RB')==None:
                                    tree.remove(vp1)
                                    tree.insert(firstvp,nextvp)
                                    base = wnl.lemmatize(vbn[0],'v')
                                    if firstvb[0] in ['was','were']:
                                        replace = Tree('VBD',[VerbMorph.find_past(base)])
                                    elif TenseChanger.issingular(obj):
                                        replace = Tree('VBZ',[VerbMorph.pluralverb(base)])
                                    else:
                                        replace = Tree('VB',[base])
                                    nextvp[vbnpos] = replace
                                    nextvp.insert(1,toobj(sub))
                                    #print(toobj(sub))
                                    nextvp.remove(pp)
                                else:
                                    if firstvb[0] in ['was','were']:
                                        firstvb[0] = 'did'
                                    elif TenseChanger.issingular(obj):
                                        firstvb[0] = 'does'
                                    else:
                                        firstvb[0] = 'do'
                                    nextvp.insert(1,toobj(sub))
                                    nextvp.remove(pp)
                else:
                    for i in tree:
                        passiveToActive(i)
            else:
                for i in tree:
                    passiveToActive(i)
        else:
            for i in tree:
                passiveToActive(i)
    else:
        return


def activeToPassive(tree):
    if isinstance(tree,Tree):
        if TenseChanger.firstword(tree) == 'let':
            return
        if tree.label()[0]=='S' and not tree.label() == 'SYM':
            firstnp = -1
            firstvp = -1
            #print(tree)
            for i in range(0,len(tree)):
                if tree[i].label()[0:2]=='NP' and firstnp < 0:
                    firstnp=i
                if tree[i].label()[0:2] == 'VP' and firstvp < 0:
                    firstvp = i
            if firstnp >= 0 and firstvp >= 0:
                vp1 = tree[firstvp]
                vb = findfirst(vp1,'VB')
                obj = findfirst(vp1,'NP')
                nextvp = findfirst(vp1,'VP')
                md = findfirst(vp1,'MD')
                rb = findfirst(vp1,'RB')
                sub = tree[firstnp]

                if (vb == None and md == None) or (obj==None and nextvp == None) or sub[0].label()=='-NONE-':
                    for i in tree:
                        activeToPassive(i)
                elif (not vb == None) and isbe(vb[0]):
                    for i in tree:
                        activeToPassive(i)
                elif not md == None:
                    if nextvp == None:
                        return
                    else:
                        vb2 = findfirst(nextvp,'VB')
                        if vb2==None:
                            for i in tree:
                                activeToPassive(i)
                            return
                        nextvp2 = findfirst(nextvp,'VP')
                        if not isbe(vb2[0]):
                            if vb2[0] in ['have', 'has', 'had'] and not nextvp2 == None:
                                obj = findfirst(nextvp2, 'NP')
                                if obj == None or nextvp2[0][0]=='been':
                                    for i in tree:
                                        activeToPassive(i)
                                    return
                                objpos = nextvp2.index(obj)
                                nextvppos = nextvp.index(nextvp2)
                                tree.remove(sub)
                                tree.insert(firstnp, tosub(obj))
                                nextvp[nextvppos] = Tree('VP', [Tree('VBN', ['been']), nextvp2])
                                nextvp2[objpos] = Tree('PP',[Tree('IN',['by']),toobj(sub)])
                            else:
                                obj = findfirst(nextvp, 'NP')
                                if obj == None:
                                    for i in tree:
                                        activeToPassive(i)
                                    return
                                objpos = nextvp.index(obj)
                                nextvppos = vp1.index(nextvp)
                                tree.remove(sub)
                                tree.insert(firstnp, tosub(obj))
                                vp1[nextvppos] = Tree('VP', [Tree('VB', ['be']), nextvp])
                                vb2 = findfirst(nextvp,'VB')
                                vb2pos = nextvp.index(vb2)
                                nextvp[vb2pos] = Tree('VBN',[VerbMorph.find_past_participle(wnl.lemmatize(vb2[0],'v'))])
                                nextvp[objpos] = Tree('PP',[Tree('IN',['by']),toobj(sub)])
                        else:
                            for i in tree:
                                activeToPassive(i)
                elif vb[0] in ['have', 'has', 'had'] and not nextvp == None:
                    vbpos = vp1.index(vb)
                    obj = findfirst(nextvp,'NP')
                    if obj == None:
                        for i in tree:
                            activeToPassive(i)
                        return
                    objpos = nextvp.index(obj)
                    nextvppos = vp1.index(nextvp)
                    tree.remove(sub)
                    tree.insert(firstnp, tosub(obj))
                    if nextvp[0][0] == 'been':
                        for i in tree:
                            activeToPassive(i)
                        return
                    vp1[nextvppos] = Tree('VP',[Tree('VBN',['been']),nextvp])
                    if vb[0] == 'had':
                        vp1[vbpos] = Tree('VBD',['had'])
                    else:
                        if TenseChanger.issingular(obj):
                            vp1[vbpos] = Tree('VBZ',['has'])
                        else:
                            vp1[vbpos] = Tree('VB',['have'])
                    nextvp[objpos] = Tree('PP',[Tree('IN',['by']),toobj(sub)])
                elif not rb == None:
                    if vb[0] in ['did','do','does']:
                        vbpos = vp1.index(vb)
                        if nextvp == None:
                            for i in tree:
                                activeToPassive(i)
                            return
                        obj = findfirst(nextvp, 'NP')
                        if obj == None:
                            for i in tree:
                                activeToPassive(i)
                            return
                        objpos = nextvp.index(obj)
                        vb2 = findfirst(nextvp, 'VB')
                        if vb2 == None:
                            return
                        vb2pos = nextvp.index(vb2)
                        tree.remove(sub)
                        tree.insert(firstnp, tosub(obj))
                        if nextvp[0][0] == 'be':
                            for i in tree:
                                activeToPassive(i)
                            return
                        if vb[0] == 'did':
                            if TenseChanger.issingular(obj):
                                vp1[vbpos] = Tree('VBD', ['was'])
                            else:
                                vp1[vbpos] = Tree('VBD', ['were'])
                        else:
                            if TenseChanger.issingular(obj):
                                vp1[vbpos] = Tree('VBZ', ['is'])
                            else:
                                vp1[vbpos] = Tree('VB', ['are'])
                        nextvp[objpos] = Tree('PP', [Tree('IN', ['by']), toobj(sub)])

                        nextvp[vb2pos] = Tree('VBN',[VerbMorph.find_past_participle(vb2[0])])
                    else:
                        for i in tree:
                            activeToPassive(i)
                elif not obj == None:
                    vbpos = vp1.index(vb)
                    objpos = vp1.index(obj)
                    sub = tree[firstnp]
                    tree.remove(sub)
                    tree.insert(firstnp, tosub(obj))
                    tree.remove(vp1)
                    objsing = TenseChanger.issingular(obj)
                    if vb.label()=='VBD':
                        if objsing:
                            newvp = Tree('VP',[Tree('VBD',['was']),vp1])
                        else:
                            newvp = Tree('VP',[Tree('VBD',['were']),vp1])
                    elif vb.label() == 'VBG':
                        newvp = Tree('VP', [Tree('VBG', ['being']), vp1])
                    else:
                        if objsing:
                            newvp = Tree('VP',[Tree('VBD',['is']),vp1])
                        else:
                            newvp = Tree('VP',[Tree('VBD',['are']),vp1])
                    tree.insert(firstvp,newvp)
                    vp1[vbpos] = Tree('VBD',[VerbMorph.find_past_participle(wnl.lemmatize(vb[0],'v'))])
                    vp1[objpos] = Tree('PP',[Tree('IN',['by']),toobj(sub)])
                else:
                    for i in tree:
                        activeToPassive(i)
            else:
                for i in tree:
                    activeToPassive(i)
        else:
            for i in tree:
                activeToPassive(i)
    else:
        return

if __name__ == "__main__":
    count=0
    for file in treebank.fileids():
        for i in treebank.parsed_sents(file):
            #if 'by' in i.leaves():
            print(i)
            activeToPassive(i)
            print(i)
            count+=1
            if count > 5:
                break
        if count > 5:
            break

