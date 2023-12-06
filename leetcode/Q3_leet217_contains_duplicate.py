
def contains_dup(mylist : list[int]) -> bool:
    myset = set()
    for ii in mylist:
        if ii in myset:
            return True
        else:
            myset.add(ii)
    return False

assert contains_dup([3, 5, 2, 6, 7, 4, 6])
assert not contains_dup([3, 5, 2, 6, 7, 4, 9])
