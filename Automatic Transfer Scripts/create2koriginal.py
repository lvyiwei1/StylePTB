import PTBdata
import random

def createtreelist():
    trees = PTBdata.getalltrees('ptb-train.txt')
    trees.extend(PTBdata.getalltrees('ptb-test.txt'))
    trees.extend(PTBdata.getalltrees('ptb-valid.txt'))
    newtrees = []
    for i in trees:
        if len(i.leaves())>12 or len(i.leaves())<5:
            continue
        if i.label()[0] != 'S':
            continue
        newtrees.append(i)
    random.seed(10)
    random.shuffle(newtrees)
    return newtrees[0:2000]


if __name__ == "__main__":
    trees = createtreelist()
    count = 0
    for i in trees:
        count += 1
        f = open("full2k/" + str(count) + ".txt", "w+")
        f.write(' '.join(i.leaves()))
        f.close()