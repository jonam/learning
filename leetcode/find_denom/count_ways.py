def count_ways(coins, S):
    # Initialize the dp array with zeros and set dp[0] to 1
    dp = [0] * (S + 1)
    dp[0] = 1
    
    # Update the dp array for each coin
    for coin in coins:
        for i in range(coin, S + 1):
            dp[i] += dp[i - coin]
            print(f"coin = {coin}, dp[{i}] = {dp[i]}")
    
    return dp[S]

# Example usage
coins = [2, 5]
S = 13
print(count_ways(coins, S))
