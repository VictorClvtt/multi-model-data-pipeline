from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.window import Window

from datetime import datetime

from src.utils.wrappers import pyspark_function_wrapper
from src.config.logger import logger

def run_silver_data_cleaning(
    spark: SparkSession,
    input_path: str,
    output_path: str,
    logger,
    read_from_bucket: bool=False,
    save_to_bucket: bool=False,
    bucket_name: str=None
) -> None:
    logger.info("Starting raw data ingestion")

    if read_from_bucket:
        df = spark.read.csv(
            f"s3a://{bucket_name}/{input_path}",
            header=True,
            inferSchema=True
        )
    else:
        df = spark.read.csv(
            input_path,
            header=True,
            inferSchema=True
        )

    logger.info("Raw data successfully read")

    # -----------------------------
    # Remove duplicate rows
    # -----------------------------
    initial_count = df.count()
    logger.info("Removing duplicate rows based on order_id")

    df = df.dropDuplicates(["order_id"])

    logger.info(
        f"Duplicate removal completed. Rows before: {initial_count}, after: {df.count()}"
    )

    # -----------------------------
    # Trim customer_name
    # -----------------------------
    logger.info("Trimming leading and trailing whitespaces from customer_name")

    df = df.withColumn(
        "customer_name",
        F.trim(F.col("customer_name"))
    )

    # -----------------------------
    # Impute unit_price using product-level average
    # -----------------------------
    logger.info("Imputing missing unit_price values using product-level averages")

    w = Window.partitionBy("product_id")

    df = df.withColumn(
        "unit_price",
        F.when(
            F.col("unit_price").isNull(),
            F.avg("unit_price").over(w)
        ).otherwise(F.col("unit_price"))
    )

    remaining_null_prices = df.where(F.col("unit_price").isNull()).count()
    logger.info(
        f"Remaining null unit_price values after imputation: {remaining_null_prices}"
    )

    # -----------------------------
    # Remove invalid quantity values
    # -----------------------------
    logger.info("Removing rows with invalid quantity values (quantity < 1)")

    df = df.where(F.col("quantity") >= 1)

    logger.info("Invalid quantity rows removed")

    # -----------------------------
    # Fix inverted dates
    # -----------------------------
    logger.info(
        "Fixing inverted dates assuming a source system error "
        "(swap between order_date and shipping_date)"
    )

    df = df.dropna(subset=["order_date"])

    df = df.withColumn("order_date_tmp", F.col("order_date"))

    df = df.withColumn(
        "order_date",
        F.when(
            (F.col("shipping_date").isNotNull()) &
            (F.col("order_date_tmp") > F.col("shipping_date")),
            F.col("shipping_date")
        ).otherwise(F.col("order_date_tmp"))
    )

    df = df.withColumn(
        "shipping_date",
        F.when(
            (F.col("shipping_date").isNotNull()) &
            (F.col("order_date_tmp") > F.col("shipping_date")),
            F.col("order_date_tmp")
        ).otherwise(F.col("shipping_date"))
    ).drop("order_date_tmp")

    logger.info("Date correction completed")

    # -----------------------------
    # Standardize emails
    # -----------------------------
    logger.info("Standardizing customer_email")

    df = (
        df.withColumn("customer_email", F.trim(F.col("customer_email")))
        .withColumn("customer_email", F.lower(F.col("customer_email")))
        .withColumn("customer_email", F.regexp_replace(F.col("customer_email"), r"\s+", ""))
        .withColumn("customer_email", F.regexp_replace(F.col("customer_email"), r"[^a-z0-9@._-]", ""))
        .withColumn("customer_email", F.regexp_replace(F.col("customer_email"), r"@+", "@"))
    )

    df = df.withColumn(
        "customer_email",
        F.when(
            F.col("customer_email").rlike(r"@[^@]+\.[^@]+"),
            F.col("customer_email")
        ).otherwise(None)
    )

    df = df.dropna(subset=["customer_email"])

    logger.info("E-mail address correction completed")

    # -----------------------------
    # Write cleaned data
    # -----------------------------
    logger.info("Writing cleaned dataset to output path")

    if save_to_bucket:
        df.write.mode("overwrite").parquet(
            f"s3a://{bucket_name}/{output_path}"
        )
    else:
        df.write.mode("overwrite").csv(output_path, header=True)

    logger.info(f"Cleaned dataset successfully written, total rows: {df.count()}")


if __name__ == "__main__":
    pyspark_function_wrapper(
        input_path="bronze/CSV/order_data.csv",
        output_path=f"silver/dt={datetime.today().date()}",
        app_name="Data Cleaning and Standardizing",
        pyspark_function=run_silver_data_cleaning,
        logger=logger,
        read_from_bucket=True,
        save_to_bucket=True
    )
