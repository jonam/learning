# https://www.youtube.com/watch?v=5WZl3MMT0Eg
def max_subarray(mylist : list[int]) -> int:
    maxSub = mylist[0]
    curSum = 0
   
    for n in mylist:
        if curSum < 0:
            curSum = 0
            print(f"n = {n}, curSum = {curSum}")
        curSum += n
        maxSub = max(maxSub, curSum)
        print(f"maxSub = {maxSub}")
    return maxSub

print(max_subarray([-2, 1, -3, 4, -1, 2, 1, -5, 4]))
print(max_subarray([-2, 1, -3, -4, -1, -2, -1, -5, -4]))
