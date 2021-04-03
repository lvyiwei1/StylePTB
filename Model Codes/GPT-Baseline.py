import transformers
import torch

if torch.cuda.is_available():
  device = torch.device('cuda:0')
else:
  device = torch.device('cpu')

gpt_tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')
gpt_model = transformers.GPT2LMHeadModel.from_pretrained('gpt2').to(device)

def padinput(inputlist, totalpad = 80):
  pads = [0]*(totalpad-len(inputlist))
  input = inputlist+pads
  mask = [1]*len(inputlist)+pads
  return input,mask

def labels(inlen,outputlist,totalpad = 80):
  pads1 = [-100]*inlen
  pads2 = [-100]*(totalpad-inlen-len(outputlist))
  #print(outputlist)
  return pads1+outputlist+pads2


def lowering(pairs, tests):
    for pair in pairs:
        for i in range(0, 2):
            pair[i] = pair[i].lower()
    for pair in tests:
        for i in range(0, 2):
            pair[i] = pair[i].lower()


def makesame(pairs, tests):
    # news = []
    for pair in pairs:
        # news.append((pair[1],pair[1]))
        pair[1] = pair[0]
    # pairs += news
    # news=[]
    for pair in tests:
        # news.append((pair[1],pair[1]))
        pair[1] = pair[0]
        # tests += news


def numpreprocess(pairs, tests):
    for pair in pairs + tests:
        for i in range(0, 2):
            rep = []
            for word in pair[i].split(' '):
                if len(word) > 0 and word[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                    rep.append("NUM")
                else:
                    rep.append(word)
            pair[i] = ' '.join(rep)


def specialnumpreprocess(pairs, tests, startloc=2):
    for pair in (pairs + tests):
        rep = []
        for word in pair[0].split(' ')[0:startloc]:
            rep.append(word)
        for word in pair[0].split(' ')[startloc:]:
            if len(word) > 0 and word[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                rep.append("NUM")
            else:
                rep.append(word)
        pair[0] = ' '.join(rep)
        rep = []
        for word in pair[1].split(' '):
            if len(word) > 0 and word[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                rep.append("NUM")
            else:
                rep.append(word)
        pair[1] = ' '.join(rep)


unkbar = 2


def preprocess(pairs, tests):
    dicts = {}
    for pair in pairs:
        for sent in pair:
            for word in sent.split(' '):
                if word not in dicts:
                    dicts[word] = 0
                dicts[word] += 1
    for pair in pairs:
        for i in range(0, 2):
            words = []
            for word in pair[i].split(' '):
                if dicts[word] <= unkbar:
                    words.append('unk')
                else:
                    words.append(word)
            pair[i] = ' '.join(words)
    for pair in tests:
        for i in range(0, 2):
            words = []
            for word in pair[i].split(' '):
                if word not in dicts or dicts[word] <= unkbar:
                    words.append('unk')
                else:
                    words.append(word)
            pair[i] = ' '.join(words)


pairs = []
tests = []
import csv

f = open('train.tsv', 'r')
ff = csv.reader(f, delimiter='\t')
limit = 10
cur = 0
for row in ff:

    # pairs.append(row)
    # """
    if row[0][0:4] in ['0 4 ', '0 5 ', '1 4 ', '2 4 ', '3 4 ']:
        pairs.append(row)
        # """
f = open('valid.tsv', 'r')
ff = csv.reader(f, delimiter='\t')
for row in ff:
    tests.append(row)

lowering(pairs, tests)
# numpreprocess(pairs,tests)
specialnumpreprocess(pairs, tests)
# makesame(pairs,tests)
# preprocess(pairs,tests)

pairsEncode = []
testsEncode = []
for i in pairs:
    pairsEncode.append((gpt_tokenizer.encode(i[0] + " <|endoftext|>"), gpt_tokenizer.encode(i[1] + " <|endoftext|>")))
for i in tests:
    testsEncode.append((gpt_tokenizer.encode(i[0] + " <|endoftext|>"), gpt_tokenizer.encode(i[1] + " <|endoftext|>")))


import torch
from torch import optim
import random

batchsize = 20
iters = 60

def train(src,trg,optim):
  padin = [padinput(l) for l in src]
  padedin = torch.LongTensor([padin[i][0] for i in range(0,len(trg))]).to(device)
  masks = torch.LongTensor([padin[i][1] for i in range(0,len(trg))]).to(device)
  label = torch.LongTensor([labels(len(src[i]),trg[i]) for i in range(0,len(trg))]).to(device)
  optim.zero_grad()
  ret = gpt_model.forward(padedin,attention_mask=masks,labels=label)
  loss=ret[0]
  loss.backward()
  optim.step()
  return loss

def valid(src,trg):
  padin = [padinput(l) for l in src]
  padedin = torch.LongTensor([padin[i][0] for i in range(0,len(trg))]).to(device)
  masks = torch.LongTensor([padin[i][1] for i in range(0,len(trg))]).to(device)
  label = torch.LongTensor([labels(len(src[i]),trg[i]) for i in range(0,len(trg))]).to(device)
  with torch.no_grad():
    ret = gpt_model.forward(padedin,attention_mask=masks,labels=label)
    loss=ret[0]
  return loss

def batchvalid(src,trg):
  validloss = 0.0
  for i in range(0,len(src)//batchsize):
    asrc=[]
    atrg=[]
    for pair in src[i*batchsize:(i+1)*batchsize]:
      asrc.append(pair)
    for pair in trg[i*batchsize:(i+1)*batchsize]:
      atrg.append(pair)
    validloss += valid(asrc,atrg)
  return validloss / (len(src)//batchsize)

def trainIter(pairs,testsrc,testtrg,optim):
  random.shuffle(pairs)
  trainloss = 0.0
  for i in range(0,len(pairs)//batchsize):
    src = []
    trg = []
    for pair in pairs[i*batchsize:(i+1)*batchsize]:
      src.append(pair[0])
      trg.append(pair[1])
    trainloss += train(src,trg,optim)
    if i%100==0:
      print("Batch "+str(i)+" of "+str(len(pairs)//batchsize))
  validloss = batchvalid(testsrc,testtrg)
  print("Trainloss: "+str((trainloss / (len(pairs)//batchsize)))+ " Valid loss: "+str(validloss))
  return validloss

tsrc =[]
ttrg = []
for pair in testsEncode:
  tsrc.append(pair[0])
  ttrg.append(pair[1])
optim = torch.optim.RMSprop(gpt_model.parameters(),lr=0.00002,weight_decay=0.015)
minloss = 100.0
for i in range(0, iters):
  if i%15 == 14:
    for param_group in optim.param_groups:
      param_group['lr'] /= 2.0
  validloss = trainIter(pairsEncode,tsrc,ttrg,optim)
  if validloss < minloss and i > 2:
    minloss = validloss
    torch.save(gpt_model,'modelIter'+str(i)+'.pt')