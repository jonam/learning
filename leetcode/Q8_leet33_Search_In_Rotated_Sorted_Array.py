
"""
l = 0, m = 2, r = 5
l = 3, m = 4, r = 5
l = 3, m = 3, r = 3
"""

# https://www.youtube.com/watch?v=U8XENwh8Oy8
def search_rot_sorted_array(target : int, mylist : list[int]) -> int:
    print()
    print(f"Searching {target} in {mylist}")
    l = 0
    r = len(mylist) - 1
    
    while l <= r:
        m = (l + r)//2
        print(f"l = {l}, m = {m}, r = {r}")
        if target == mylist[m]:
            return m

        # left sorted portion
        if mylist[l] <= mylist[m]:
            if target > mylist[m] or target < mylist[l]:
                l = m + 1
            else:
                r = m - 1
        else: # right sorted portion
            if target < mylist[m] or target > mylist[r]:
                r = m - 1
            else:
                l = m + 1
         
    return -1

assert search_rot_sorted_array(0, [5, 6, 7, 0, 1, 4]) == 3
assert search_rot_sorted_array(6, [5, 6, 7, 0, 1, 4]) == 1
assert search_rot_sorted_array(4, [8, 9, 10, 12, 14, 0, 1, 4, 5, 6, 7]) == 7
