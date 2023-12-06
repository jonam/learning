"""
         1   2   3   4

Prefix   1   2   6   24    (------>)

Postfix  24  24  12  4     (<------)

Output   24  12  8   6
"""

# https://www.youtube.com/watch?v=bNvIQI2wAjk
def prod_array(mylist : list[int]) -> list[int]:
    mylen = len(mylist)
    output = [1]*mylen
    output[0] = 1
    fix = mylist[0]
    for rind, ii in enumerate(mylist[1:]):
        output[rind+1] = fix
        fix *= ii
    fix = mylist[mylen-1]
    for rind, ii in enumerate(mylist[:-1]):
        output[mylen - rind - 2] *= fix
        fix *= mylist[mylen - rind - 2]
        print(f"fix = {fix}")
    return output

assert(prod_array([1, 2, 3, 4]) == ([24, 12, 8, 6])
