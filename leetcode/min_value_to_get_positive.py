"""
[-3,2,-3,4,2]
[-3,-1,-4,0,2]
[1, 2]
[1, 3]
[-3,6,2,5,8,6]
[-3,3,5,10,18,24]
"""
def min_value(nums):
    prefix = [nums[0]] * len(nums)
    minPrefix = nums[0]
    for ii in range(1, len(nums)):
        prefix[ii] = nums[ii] + prefix[ii-1]
        minPrefix = min(prefix[ii], minPrefix) 
  
    startValue = 1 - minPrefix if minPrefix < 0 else 1
    return startValue

assert min_value([-3,2,-3,4,2]) == 5
assert min_value([1, 2]) == 1
assert min_value([-3,6,2,5,8,6]) == 4
