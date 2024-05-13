-- https://leetcode.com/problems/market-analysis-i/
-- Write a solution to find for each user, the join date and the number of orders they made as a buyer in 2019.
-- Return the result table in any order.

SELECT u.user_id as buyer_id, join_date, COUNT(o.order_id) AS orders_in_2019
FROM Users u
LEFT JOIN Orders o ON u.user_id = o.buyer_id AND YEAR(order_date)='2019'
GROUP BY u.user_id
