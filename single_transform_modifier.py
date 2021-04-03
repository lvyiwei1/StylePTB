import sys

def same_is(s1,s2):
    for i in range(len(s1)):
        if not s1[i]==s2[i]:
            if s1[i]=='\'' and s2[i]=='i':
                continue
            else:
                return False
    return True

f=open('fulldata.h16','r')

lines=f.readlines()

f.close()

ne=[]

for i in range(len(lines)//3):
    if not same_is(lines[3*i+1],lines[3*i+2]):
        ne.append(lines[3*i])
        ne.append(lines[3 * i+1])
        ne.append(lines[3 * i+2])


f = open('fulldatatmp.h16', 'w+')
for line in ne:
    f.write(line)
f.close()



