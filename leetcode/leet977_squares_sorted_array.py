def sortedSquares(nums):
    x = [ii*ii for ii in nums]
    x.sort()
    return x

print(sortedSquares([-4,-1,0,3,10]))
