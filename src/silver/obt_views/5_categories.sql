CREATE OR REPLACE VIEW analytics.vw_sales_by_category AS
SELECT
    category,
    COUNT(DISTINCT order_id) AS total_orders,
    SUM(quantity) AS total_items,
    SUM(revenue) AS total_revenue
FROM analytics.vw_obt_base
GROUP BY category;