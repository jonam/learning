
def two_sum(mysum : int, mylist : list[int]) -> tuple:
    mydict = {}
    for rind, ii in enumerate(mylist):
        if (mysum - ii) in mydict:
            return (mydict[mysum-ii], rind) 
        mydict[ii] = rind
             

assert two_sum(4, [1, 4, 3, 2, 7]) == (0, 2)
assert two_sum(10, [1, 4, 3, 2, 7]) == (2, 4)
assert two_sum(5, [1, 4, 3, 2, 7]) == (0, 1)
