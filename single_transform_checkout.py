import sys

alltrans=['IAD','AEM','VEM','NSR','ASR','VSR','NAR','VAR','AAR','ARR','PPR','SBR','MFS','LFS','TFU','TPA','TPR','PTA','ATP','PFB','PBF']

if sys.argv[1] not in alltrans:
    print('Error: Transformation code not found!')
    exit(1)

f=open('fulldata.h16','r')

lines=f.readlines()

pairs=[]

for i in range(len(lines)):
    if lines[i][0:3] == sys.argv[1]:
        pairs.append([lines[i+1][:-1],lines[i+2][:-1]])

print(len(pairs))

validsplit=len(pairs)//20
testsplit=len(pairs)//10
if validsplit < 1:
    validsplit=1
if testsplit < 2:
    testsplit=2


import os
os.mkdir(sys.argv[1])
f1=open(sys.argv[1]+'/valid.tsv','w+')
f2=open(sys.argv[1]+'/test.tsv','w+')
f3=open(sys.argv[1]+'/train.tsv','w+')
for i in range(len(pairs)):
    if i < validsplit:
        f1.write(pairs[i][0]+'\t'+pairs[i][1]+'\n')
    elif i < testsplit:
        f2.write(pairs[i][0]+'\t'+pairs[i][1]+'\n')
    else:
        f3.write(pairs[i][0]+'\t'+pairs[i][1]+'\n')


f1.close()
f2.close()
f3.close()
