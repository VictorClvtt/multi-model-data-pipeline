CREATE OR REPLACE VIEW analytics.vw_payment_analysis AS
SELECT
    payment_method,
    COUNT(DISTINCT order_id) AS total_orders,
    SUM(revenue) AS total_revenue
FROM analytics.vw_obt_base
GROUP BY payment_method;