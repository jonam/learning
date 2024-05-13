/*
 https://leetcode.com/problems/human-traffic-of-stadium/description/?envType=problem-list-v2&envId=e97a9e5m
 Write a solution to display the records with three or more rows with consecutive id's, and the number of people is greater than or equal to 100 for each.
 Return the result table ordered by visit_date in ascending order.

 Common Table Expression (CTE) - q1:
 This is a temporary result set (named q1) that is defined within the execution scope of a single SELECT, INSERT, UPDATE, or DELETE statement. Here's what happens inside q1:

select *, id - row_number() over() as id_diff: This line selects all columns from the stadium table, and it calculates a new column called id_diff. id_diff is the result of subtracting the row_number() (a sequential integer assigned to rows in the order of the SQL's execution) from the id column of the stadium table.
 from stadium where people > 99: The query is filtering to only include rows from stadium where the people column exceeds 99.
 The purpose of calculating id_diff is to identify sequences of records. If id is consecutive for entries with more than 99 people, id_diff will remain the same because each subtraction of row_number() from id will cancel out the increment.
 Main Query:
 
 select id, visit_date, people from q1: After setting up the CTE, the main query selects id, visit_date, and people from q1.
 where id_diff in (select id_diff from q1 group by id_diff having count(*) > 2): This line filters the results to only include those where their id_diff appears in a subquery that groups q1 by id_diff and filters those groups to ones having more than two rows. Essentially, it's looking for sequences where there are at least three consecutive records with people greater than 99.
 order by visit_date: Finally, the results are ordered by the visit_date column.
*/

with q1 as (
select *, id - row_number() over() as id_diff
from stadium
where people > 99
)
select id, visit_date, people
from q1
where id_diff in (select id_diff from q1 group by id_diff having count(*) > 2)
order by visit_date
