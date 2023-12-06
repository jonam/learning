
# Adjacent numbers multipled can become alternative -ve or +ve
# depending upon odd number or even number of negative numbers
# And hence curMax may become curMin just with a negative sign
# and vice versa. So we use dynamic programming here.
# Also special case if all are positive, we just multiply all of them
# Another special case is if a number is 0, then we reset curMin and curMax 
# to be 1.

# https://www.youtube.com/watch?v=lXVy6YWFcRM
def max_prod_subarray(mylist : list[int]) -> int:
    res = max(mylist)
    curMin, curMax = 1, 1
  
    for n in mylist:
        if n == 0:
            curMin, curMax = 1, 1
            continue

        tmp = curMax * n  # since curMax gets altered below
        curMax = max(n * curMax, n * curMin, n)
        curMin = min(tmp, n * curMin, n)

        res = max(res, curMax, curMin)

    return res

assert max_prod_subarray([1, 2, -2, -1]) == 4
assert max_prod_subarray([1, 3, -2, 5, -1, 2, -5]) == 60
