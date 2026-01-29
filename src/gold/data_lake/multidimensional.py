from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import logging

from src.utils.wrappers import pyspark_function_wrapper

def run_gold_data_modeling(spark: SparkSession, input_path: str, output_path: str, logger):
    logger.info("Reading clean data")
    df = spark.read.csv(input_path, header=True, inferSchema=True)

    logger.info("Starting gold layer data modeling for analytics")

    ##############
    # FACT TABLE #
    ##############
    fact_order = (
        df.select(
            "order_id",
            "order_date",
            "shipping_date",
            "customer_id",
            "product_id",
            "payment_method",
            "status",
            "quantity",
            "unit_price"
        )
        .withColumn(
            "order_date_sk",
            F.sha2(F.col("order_date").cast("date").cast("string"), 256)
        )
        .withColumn(
            "shipping_date_sk",
            F.sha2(F.col("shipping_date").cast("date").cast("string"), 256)
        )
        .withColumn(
            "customer_sk",
            F.sha2(F.col("customer_id").cast("string"), 256)
        )
        .withColumn(
            "product_sk",
            F.sha2(F.col("product_id").cast("string"), 256)
        )
        .withColumn(
            "payment_method_sk",
            F.sha2(F.col("payment_method").cast("string"), 256)
        )
        .withColumn(
            "order_status_sk",
            F.sha2(F.col("status").cast("string"), 256)
        )
        .withColumn(
            "total_amount",
            F.col("quantity") * F.col("unit_price")
        )
        .select(
            "order_id",
            "order_date_sk",
            "shipping_date_sk",
            "customer_sk",
            "product_sk",
            "payment_method_sk",
            "order_status_sk",
            "quantity",
            "unit_price",
            "total_amount"
        )
    )

    ############
    # DIM DATE #
    ############
    dim_date = (
        df.select(F.col("order_date").cast("date").alias("full_date"))
        .union(
            df.select(F.col("shipping_date").cast("date").alias("full_date"))
        )
        .filter(F.col("full_date").isNotNull())
        .distinct()
        .withColumn("day", F.dayofmonth("full_date"))
        .withColumn("month", F.month("full_date"))
        .withColumn("month_name", F.date_format("full_date", "MMMM"))
        .withColumn("quarter", F.quarter("full_date"))
        .withColumn("year", F.year("full_date"))
        .withColumn("day_of_week", F.date_format("full_date", "EEEE"))
        .withColumn(
            "is_weekend",
            F.when(F.dayofweek("full_date").isin(1, 7), F.lit(True))
            .otherwise(F.lit(False))
        )
        .withColumn(
            "date_sk",
            F.sha2(F.col("full_date").cast("string"), 256)
        )
    )

    ####################
    # DIM ORDER STATUS #
    ####################
    dim_order_status = (
        df.select("status")
        .filter(F.col("status").isNotNull())
        .distinct()
        .withColumn(
            "order_status_sk",
            F.sha2(F.col("status").cast("string"), 256)
        )
    )

    ######################
    # DIM PAYMENT METHOD #
    ######################
    dim_payment_method = (
        df.select("payment_method")
        .filter(F.col("payment_method").isNotNull())
        .distinct()
        .withColumn(
            "payment_method_sk",
            F.sha2(F.col("payment_method").cast("string"), 256)
        )
    )

    ###############
    # DIM PRODUCT #
    ###############
    dim_product = (
        df.select("product_id", "product_name", "category", "supplier")
        .filter(F.col("product_id").isNotNull())
        .distinct()
        .withColumn(
            "product_sk",
            F.sha2(F.col("product_id").cast("string"), 256)
        )
    )

    ################
    # DIM CUSTOMER #
    ################
    dim_customer = (
        df.select(
            "customer_id",
            "customer_name",
            "customer_email",
            "customer_city",
            "customer_state"
        )
        .filter(F.col("customer_id").isNotNull())
        .distinct()
        .withColumn(
            "customer_sk",
            F.sha2(F.col("customer_id").cast("string"), 256)
        )
        .withColumn(
            "location_sk",
            F.sha2(
                F.concat_ws(
                    "|",
                    F.col("customer_city"),
                    F.col("customer_state")
                ),
                256
            )
        )
        .drop("customer_city", "customer_state")
    )

    ################
    # DIM LOCATION #
    ################
    dim_location = (
        df.select("customer_city", "customer_state")
        .filter(
            F.col("customer_city").isNotNull() &
            F.col("customer_state").isNotNull()
        )
        .distinct()
        .withColumn(
            "location_sk",
            F.sha2(
                F.concat_ws(
                    "|",
                    F.col("customer_city"),
                    F.col("customer_state")
                ),
                256
            )
        )
        .withColumnRenamed("customer_city", "city")
        .withColumnRenamed("customer_state", "state")
    )

    #####################
    # WRITE GOLD TABLES #
    #####################
    logger.info(f"Recording gold schema tables to {output_path}")

    fact_order.write.parquet(f"{output_path}/fact_order", mode="overwrite")
    dim_date.write.parquet(f"{output_path}/dim_date", mode="overwrite")
    dim_order_status.write.parquet(f"{output_path}/dim_order_status", mode="overwrite")
    dim_payment_method.write.parquet(f"{output_path}/dim_payment_method", mode="overwrite")
    dim_product.write.parquet(f"{output_path}/dim_product", mode="overwrite")
    dim_customer.write.parquet(f"{output_path}/dim_customer", mode="overwrite")
    dim_location.write.parquet(f"{output_path}/dim_location", mode="overwrite")

    logger.info("Gold layer data modeling finished successfully")

if __name__ == "__main__":
    # Logger configuration
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M"
    )

    logger = logging.getLogger(__name__)

    pyspark_function_wrapper(
        input_path="./data/silver/clean_orders/",
        output_path="./data/gold",
        app_name="Multi Dimensional Modelling",
        pyspark_function=run_gold_data_modeling,
        logger=logger
    )