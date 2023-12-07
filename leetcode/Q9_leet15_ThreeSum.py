
# https://www.youtube.com/watch?v=jzZsG8n2R9A
# [-3 -3 -1 1 2 3 5]
def threeSum(mylist : list[int]) -> list[int]:
    res = []
    mylist.sort()

    if len(mylist) == 0 or mylist[0] > 0:
        return res

    for i, n in enumerate(mylist):
        if i > 0 and n == mylist[i-1]:
            continue

        print(f"i = {i}")

        l, r = i + 1, len(mylist) - 1
        while l < r:
            sum3 = n + mylist[l] + mylist[r]
            if sum3 > 0:
                r -= 1
            elif sum3 < 0:
                l += 1
            else:
                res.append([n, mylist[l], mylist[r]])
                l += 1
                while mylist[l] == mylist[l-1] and l < r:
                    l += 1
    return res

#assert threeSum([-3, 3, 4, -3, 1, 2]) == [[-3, 1, 2]]
#assert threeSum([-3, -3, -1, 1, 2, 3, 5]) == [[-3, 1, 2]]
assert threeSum([1, 2, 3, 5]) == []
