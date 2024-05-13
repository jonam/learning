-- https://leetcode.com/problems/article-views-i/description/?envType=problem-list-v2&envId=e97a9e5m
-- Write a solution to find all the authors that viewed at least one of their own articles.
-- Return the result table sorted by id in ascending order.

select distinct author_id as id from Views
where author_id = viewer_id 
order by id;
