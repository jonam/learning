def sortedSquares(nums):
    rsp = [0] * len(nums)
    i = 0
    j = k = len(nums) - 1
    while k >= 0:
        if abs(nums[i]) < abs(nums[j]):
            rsp[k] = nums[j] * nums[j]
            j -= 1
        else:
            rsp[k] = nums[i] * nums[i]
            i += 1
        k -= 1
    return rsp

assert sortedSquares([-4,-1,0,3,10]) == [0, 1, 9, 16, 100]

