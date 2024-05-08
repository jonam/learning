
def most_water(height: list[int]) -> int:
    l, r = (0, len(height) -1)
    res = 0
    while l < r:
        area = (r - l) * min(height[l], height[r])
        res = max(res, area)
       
        if height[l] < height[r]:
            l += 1
        else:
            r -= 1

    return res 

print(most_water([1, 8, 6, 2, 5, 4, 8, 3, 7]))
