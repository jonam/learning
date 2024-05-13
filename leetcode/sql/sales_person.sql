-- https://leetcode.com/problems/sales-person/description/?envType=problem-list-v2&envId=e97a9e5m
-- Write a solution to find the names of all the salespersons who did not have any orders related to the company with the name "RED".
-- Return the result table in any order.

SELECT name
FROM SalesPerson AS s
WHERE NOT EXISTS (
    SELECT 1
    FROM Orders AS o
    WHERE s.sales_id = o.sales_id  
        AND com_id = (SELECT com_id FROM Company WHERE name = 'RED')
)
