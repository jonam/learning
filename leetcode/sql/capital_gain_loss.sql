-- https://leetcode.com/problems/capital-gainloss/description/
-- Write a solution to report the Capital gain/loss for each stock.
-- The Capital gain/loss of a stock is the total gain or loss after buying and selling the stock one or many times.
-- Return the result table in any order.

select 
       stock_name,  
       sum(case when operation = 'Sell' then price else NULL end )- sum(case when operation = 'Buy' then price else NULL end) as capital_gain_loss
       from Stocks 
       group by stock_name ; 
