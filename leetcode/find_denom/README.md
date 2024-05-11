Given the set of coins, find the number of ways to achieve a target x.

But this is not the question.

The above question can be solved using Recursion + memoization or tabulation.

The question is

In the tabulation method we form a coins dp something like [1,0,1,0,1,1,2]

Given this dp array, we have to return the set of coins.

Eg:

Input: [1,0,1,0,1,1,2,1,2,2,3] Output: [2,5,6]

What does this dp array specify is that,

Each index is a target, and the value is the number of ways we can make the target using the unknown set of denominations(which is what we have to find out), so basically we have some targets ranging from 0 to x and each value is number of ways to form each target. With this, we have to traceback using what set of denominations we can achieve this dp set.

Has anyone seen this question before and do you what category it falls under is it medium or hard and any insights...
