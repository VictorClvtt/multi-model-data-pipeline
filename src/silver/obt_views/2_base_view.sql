CREATE OR REPLACE VIEW analytics.vw_obt_base AS
SELECT
    order_id,
    order_date,
    shipping_date,
    dt,
    status,
    quantity,
    unit_price,
    quantity * unit_price AS revenue,
    payment_method,
    customer_id,
    customer_name,
    customer_email,
    customer_city,
    customer_state,
    product_id,
    product_name,
    category,
    supplier
FROM obt;