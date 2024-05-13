-- https://leetcode.com/problems/tree-node/description/?envType=problem-list-v2&envId=e97a9e5m
-- Write a solution to report the type of each node in the tree.
-- Return the result table in any order.

SELECT id, 
    CASE
        WHEN p_id IS NULL THEN 'Root'
        WHEN id IN (
            SELECT p_id FROM Tree
        ) THEN 'Inner'
        ELSE 'Leaf' 
    END AS type
FROM Tree
