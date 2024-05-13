-- https://leetcode.com/problems/user-activity-for-the-past-30-days-i/description/?envType=problem-list-v2&envId=e97a9e5m
-- Write a solution to find the daily active user count for a period of 30 days ending 2019-07-27 inclusively. A user was active on someday if they made at least one activity on that day.
-- Return the result table in any order.

SELECT activity_date AS day, COUNT(DISTINCT user_id) AS active_users
FROM Activity
WHERE (activity_date > "2019-06-27" AND activity_date <= "2019-07-27")
GROUP BY activity_date;
