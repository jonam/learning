"""
Given an integer array nums, find the number of ways to split the array into two parts so that the first section has a sum greater than or equal to the sum of the second section. The second section should have at least one number.

A brute force approach would be to iterate over each index i from 0 until nums.length - 1. For each index, iterate from 0 to i to find the sum of the left section, and then iterate from i + 1 until the end of the array to find the sum of the right section. This algorithm would have a time complexity of O(n^2).

If we build a prefix sum first, then iterate over each index, we can calculate the sums of the left and right sections in O(1), which would improve the time complexity to O( n).
"""

def waysToSplitArray(nums: list[int]) -> int:
    ans = left_section = 0
    total = sum(nums)

    for i in range(len(nums) - 1):
        left_section += nums[i]
        right_section = total - left_section
        if left_section >= right_section:
            ans += 1

    return ans

assert waysToSplitArray([10, 4, -8, 7]) == 2
