dict = {}
with open("irregular_verbs.txt","r") as f:
    lines = f.readlines()
    for line in lines:
        tenses = line.split("\t")
        dict[tenses[0]]=tenses

def is_vowel(c):
    return c=='a' or c=='e' or c=='i' or c=='o' or c=='u'

def is_conso(c):
    return (not is_vowel(c)) and (not c=='y') and (not c=='h') and (not c=='r') and (not c == 'l') and (not c == 'w')

def regular_past(word):
    if word[-1] == 'e':
        return word + 'd'
    elif word[-1] == 'y' and is_conso(word[-2]):
        return word[:-1]+"ied"
    elif len(word)>2 and is_conso(word[-1]) and is_vowel(word[-2]) and not is_vowel(word[-3]):
        return word + word[-1] + 'ed'
    else:
        return word+'ed'

def find_past_participle(word):
    if word.lower() == '\'s':
        return 'had'
    if word in dict:
        z = dict[word][2]
        if z[-1] == '\n':
            return z[:-1]
        else:
            return z
    else:
        return regular_past(word)

def find_past(word):
    if word=='\'m' or word=='am':
        return 'was'
    if word == '\'s':
        return 'had'
    if word == 'is':
        return 'was'
    elif word == 'are':
        return 'were'
    elif word in dict:
        return dict[word][1]
    else:
        return regular_past(word)

def pluralverb(word):
    if word == 'have':
        return 'has'
    if word == 'do':
        return 'does'
    if word == 'is' or word == 'are':
        return 'is'
    if word[-1] == 'y' and not is_vowel(word[-2]):
        return word[:-1]+'ies'
    elif word[-1] == 's' or word[-1] == 'z' or word[-1] == 'x':
        return word + "es"
    else:
        return word + "s"