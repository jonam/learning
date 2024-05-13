
"""
7 4 3 9 1 8 5 2 6
7 11 14 23 24 32 37 39 45
"""

def kradius_avg(nums, k):
    outarr = [-1] * len(nums)
    prefix = [nums[0]] * len(nums)
    for i in range(1, len(nums)):
        prefix[i] = prefix[i-1] + nums[i]

    for i in range(k, len(nums)-k):
        outarr[i] = prefix[i+k] if i == k else (prefix[i+k] - prefix[i-k-1])
        outarr[i] //= (2*k + 1)

    return outarr

assert kradius_avg([7,4,3,9,1,8,5,2,6],3) == [-1,-1,-1,5,4,4,-1,-1,-1]
assert kradius_avg([10000],0) == [10000]
assert kradius_avg([0],10000) == [-1]
