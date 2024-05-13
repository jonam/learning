"""
Given an array nums. We define a running sum of an array as runningSum[i] = sum(nums[0]…nums[i]).

Return the running sum of nums.
"""

def runningSum(nums: list[int]) -> int:
    n = len(nums)
     
    prefix = [nums[0]]
    for i in range(1, n):
        prefix.append(nums[i] + prefix[-1])

    return prefix

assert runningSum([1,2,3,4]) == [1,3,6,10]
