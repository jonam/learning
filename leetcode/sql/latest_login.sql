-- https://leetcode.com/problems/the-latest-login-in-2020/description/
-- Write a solution to report the latest login for all users in the year 2020. Do not include the users who did not login in 2020.
-- Return the result table in any order.

SELECT
    user_id,
    MAX(time_stamp) AS last_stamp #obtaining latest login for all users
FROM Logins
WHERE YEAR(time_stamp) = 2020 #filtering for login dates with year 2020 in timestamp
GROUP BY user_id;
