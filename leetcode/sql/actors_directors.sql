-- https://leetcode.com/problems/actors-and-directors-who-cooperated-at-least-three-times/?envType=problem-list-v2&envId=e97a9e5m
-- Actors and Directors Who Cooperated At Least Three Times
-- Write a solution to find all the pairs (actor_id, director_id) where the actor has cooperated with the director at least three times.
-- Return the result table in any order.

select actor_id, director_id 
from(
select actor_id,director_id, 
count(timestamp) as cooperated 
from ActorDirector 
group by actor_id,director_id) 
table1
where cooperated>=3;
