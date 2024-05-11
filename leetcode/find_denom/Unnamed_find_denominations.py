def valid_ways(coins : list[int], S: int, ways: int) -> bool:
    # Initialize the dp array with zeros and set dp[0] to 1
    dp = [0] * (S + 1)
    dp[0] = 1

    if S == 0:
        return ways == 1

    #print(f"coin = {coins}, S = {S}, ways = {ways}")

    # Update the dp array for each coin
    for coin in coins:
        for i in range(coin, S + 1):
            dp[i] += dp[i - coin]
            #print(f"coin = {coin}, dp[{i}] = {dp[i]}")

    #print(f"dp[{S}] = {dp[S]}, {ways} -1 = {ways-1}")
    return dp[S] != (ways - 1)
    
def find_denom(dp : list[int]) -> list[int]:
    dpout = []
    for ii in range(len(dp)):
        if dp[ii] > 0:
            if not valid_ways(dpout, ii, dp[ii]):
                dpout.append(ii)
    return dpout

print(find_denom([1,0,1,0,1,1,2,1,2,2,3]))
