
# https://www.youtube.com/watch?v=nIVW4P8b1VA

def find_min_rot_sorted(mylist : list[int]) -> int:
    l = 0
    r = len(mylist) - 1
    res = mylist[0]

    while l <= r:
        # if array is already sorted
        if mylist[l] < mylist[r]:
            res = min(res, mylist[l])
            break

        m = (l + r)//2
        res = min(res, mylist[m])
        if mylist[m] >= mylist[l]:
            l = m + 1
        else:
            r = m - 1

    return res

assert find_min_rot_sorted([5, 6, 7, 0, 1, 4]) == 0
