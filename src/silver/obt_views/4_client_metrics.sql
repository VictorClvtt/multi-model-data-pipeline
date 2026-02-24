CREATE OR REPLACE VIEW analytics.vw_customer_metrics AS
SELECT
    customer_id,
    customer_name,
    customer_city,
    customer_state,
    COUNT(DISTINCT order_id) AS total_orders,
    SUM(revenue) AS lifetime_value,
    MAX(order_date) AS last_purchase
FROM analytics.vw_obt_base
GROUP BY 1,2,3,4;