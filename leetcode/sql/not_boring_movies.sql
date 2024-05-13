-- https://leetcode.com/problems/not-boring-movies/description/?envType=problem-list-v2&envId=e97a9e5m
-- Write a solution to report the movies with an odd-numbered ID and a description that is not "boring".
-- Return the result table ordered by rating in descending order.

SELECT *
FROM Cinema
WHERE description <> 'boring' and MOD(id,2) <> 0
ORDER BY rating DESC;
