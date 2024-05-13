-- https://leetcode.com/problems/group-sold-products-by-the-date/description/?envType=problem-list-v2&envId=e97a9e5m
-- Write a solution to find for each date the number of different products sold and their names.
-- The sold products names for each date should be sorted lexicographically.
-- Return the result table ordered by sell_date.

SELECT sell_date, count(DISTINCT product) as num_sold,
 GROUP_CONCAT(DISTINCT product ORDER BY product ASC separator ',') as products
FROM Activities
GROUP BY sell_date ORDER BY sell_date ASC
