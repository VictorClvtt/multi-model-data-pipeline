from pyspark.sql import SparkSession
import pyspark.sql.functions as F

from src.utils.wrappers import pyspark_function_wrapper
from src.config.logger import logger


def run_gold_data_mirroring(
    spark: SparkSession,
    input_path: str,
    output_path: str,
    logger,
    read_from_bucket: bool=False,
    save_to_bucket: bool=False,
    bucket_name: str=None
):

    logger.info("Reading gold tables")

    base_bucket_path = f"s3a://{bucket_name}/{input_path}"

    # =========================
    # READ GOLD TABLES
    # =========================

    fact_order = spark.read.parquet(f"{base_bucket_path}/fact_order")
    dim_date = spark.read.parquet(f"{base_bucket_path}/dim_date")
    dim_order_status = spark.read.parquet(f"{base_bucket_path}/dim_order_status")
    dim_payment_method = spark.read.parquet(f"{base_bucket_path}/dim_payment_method")
    dim_product = spark.read.parquet(f"{base_bucket_path}/dim_product")
    dim_customer = spark.read.parquet(f"{base_bucket_path}/dim_customer")
    dim_location = spark.read.parquet(f"{base_bucket_path}/dim_location")

    logger.info("Gold tables successfully read")

    # =========================
    # JDBC CONFIG
    # =========================

    logger.info("Writing gold tables to the Data Warehouse")

    jdbc_url = "jdbc:postgresql://localhost:5432/dw"

    connection_properties = {
        "user": "dw_user",
        "password": "dw_password",
        "driver": "org.postgresql.Driver"
    }

    # =========================
    # TABLE MAP
    # =========================

    tables = {
        "fact_order": fact_order,
        "dim_date": dim_date,
        "dim_order_status": dim_order_status,
        "dim_payment_method": dim_payment_method,
        "dim_product": dim_product,
        "dim_customer": dim_customer,
        "dim_location": dim_location,
    }

    # =========================
    # WRITE LOOP
    # =========================

    for table_name, df in tables.items():

        logger.info(f"Writing table {table_name}")

        (
            df.repartition(4)  # controla paralelismo
            .write
            .format("jdbc")
            .option("url", jdbc_url)
            .option("dbtable", table_name)
            .option("batchsize", 10000)
            .option("numPartitions", 4)
            .options(**connection_properties)
            .mode("append")   # NÃO usar overwrite em produção
            .save()
        )

    logger.info("All gold tables successfully written to DW")


if __name__ == "__main__":
    pyspark_function_wrapper(
        app_name="Gold Layer Data Mirroring",
        pyspark_function=run_gold_data_mirroring,
        logger=logger,
        input_path="gold",
        read_from_bucket=True
    )