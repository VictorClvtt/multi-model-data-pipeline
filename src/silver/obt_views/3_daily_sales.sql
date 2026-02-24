CREATE OR REPLACE VIEW analytics.vw_sales_daily AS
SELECT
    dt,
    COUNT(DISTINCT order_id) AS total_orders,
    SUM(quantity) AS total_items,
    SUM(revenue) AS total_revenue,
    AVG(revenue) AS avg_ticket
FROM analytics.vw_obt_base
GROUP BY dt;