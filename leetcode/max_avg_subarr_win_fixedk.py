
def max_avg_win(nums, k):
    left = curr = 0
    right = k

    for i in range(k):
        curr += nums[i]

    max_subarr = (left, right)
    max_curr = curr
  
    for right in range(k, len(nums)):
        curr += nums[right]
        curr -= nums[left]
        left += 1
        if curr > max_curr:
            max_subarr = (left, right)
            max_curr = curr

    #print(nums[max_subarr[0]:max_subarr[1]+1])
    return max_curr/k

#assert max_avg_win([1,12,-5,-6,50,3], 4) == 12.75
assert max_avg_win([5], 1) == 5
print(max_avg_win([1,12,-5,-6,50,3], 4))
