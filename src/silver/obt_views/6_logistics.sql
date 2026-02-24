CREATE OR REPLACE VIEW analytics.vw_shipping_performance AS
SELECT
    AVG(shipping_date - order_date) AS avg_shipping_time,
    MAX(shipping_date - order_date) AS max_shipping_time
FROM analytics.vw_obt_base
WHERE shipping_date IS NOT NULL;