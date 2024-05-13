-- https://leetcode.com/problems/top-travellers/description/
-- Write a solution to report the distance traveled by each user.
-- Return the result table ordered by travelled_distance in descending order, if two or more users traveled the same distance, order them by their name in ascending order.

select name,SUM(case when distance is null then 0 else distance end) as travelled_distance 
from users 
left join rides
on users.id=rides.user_id
group by user_id,name
order by travelled_distance desc,name
