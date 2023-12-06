import sys

def best_time(mylist : list[int]) -> tuple[int]:
    mydict = {}
    cmin = 10000
    cmax = 0
    mind = 10000
    maxd = 0
    for rind, ii in enumerate(mylist):
        if (ii < cmin):
            cmin = ii
            mind = rind+1
        elif (ii > cmax):
            cmax = ii
            maxd = rind+1
    return (mind, maxd)

assert best_time([7, 1, 9, 3, 6, 4]) == (2, 3)
assert best_time([7, 1, 4, 3, 6, 4]) == (2, 5)
        
