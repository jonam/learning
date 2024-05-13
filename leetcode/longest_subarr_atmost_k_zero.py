def find_length(nums, k):
    # curr is the current number of zeros in the window
    left = curr = ans = 0 
    for right in range(len(nums)):
        if nums[right] == 0:
            curr += 1
        while curr > k:
            if nums[left] == 0:
                curr -= 1
            left += 1
        ans = max(ans, right - left + 1)
    
    return ans

assert find_length([1,1,1,0,0,0,1,1,1,1,0], 2) == 6
assert find_length([0,0,1,1,0,0,1,1,1,0,1,1,0,0,0,1,1,1,1], 3) == 10
