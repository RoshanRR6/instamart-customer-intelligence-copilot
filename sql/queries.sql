-- Q1 — WHERE then HAVING
SELECT customer_id, SUM(order_amount) AS spend_over_500_inr
FROM orders
WHERE status = 'completed' AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000
ORDER BY spend_over_500_inr DESC;

-- Q2 — three-table join
SELECT c.city, p.category, ROUND(SUM(o.order_amount), 2) AS revenue_inr
FROM orders o
JOIN customers c ON c.customer_id = o.customer_id
JOIN products p ON p.product_id = o.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY c.city, p.category;

-- Q3 — scalar subquery
SELECT product_id, product_name, price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products)
ORDER BY price_inr DESC, product_id;

-- Q4 — LEFT JOIN
SELECT c.customer_id, c.city
FROM customers c
LEFT JOIN orders o ON o.customer_id = c.customer_id AND o.status = 'completed'
WHERE o.order_id IS NULL
ORDER BY c.customer_id;

-- Q5 — common table expressions
WITH customer_spend AS (
    SELECT customer_id, SUM(order_amount) AS total_spend
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
), average_spend AS (
    SELECT AVG(total_spend) AS avg_spend FROM customer_spend
)
SELECT cs.customer_id, cs.total_spend, av.avg_spend
FROM customer_spend cs CROSS JOIN average_spend av
WHERE cs.total_spend > av.avg_spend
ORDER BY cs.total_spend DESC;

-- Q6 — ranking window functions
WITH customer_spend AS (
    SELECT c.customer_id, c.city, SUM(o.order_amount) AS total_spend
    FROM customers c
    JOIN orders o ON o.customer_id = c.customer_id
    WHERE o.status = 'completed'
    GROUP BY c.customer_id, c.city
)
SELECT customer_id, city, ROUND(total_spend, 2) AS total_spend_inr,
       RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS spend_rank,
       DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS dense_spend_rank,
       ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC) AS spend_row_number
FROM customer_spend
ORDER BY city, spend_rank, customer_id;

-- Q7 — running total
WITH daily_revenue AS (
    SELECT SUBSTR(order_ts, 1, 10) AS day, ROUND(SUM(order_amount), 2) AS revenue_inr
    FROM orders
    WHERE status = 'completed'
    GROUP BY SUBSTR(order_ts, 1, 10)
)
SELECT day, revenue_inr,
       ROUND(SUM(revenue_inr) OVER (ORDER BY day ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW), 2) AS cumulative_revenue_inr
FROM daily_revenue
ORDER BY day;
