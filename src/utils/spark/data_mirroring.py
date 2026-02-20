from pyspark.sql import SparkSession

def data_mirroring(
    spark: SparkSession,
    input_path: str,
    logger,
    table_names: list[str],
    output_path: str | None = None,
    read_from_bucket: bool = False,
    save_to_bucket: bool = False,
    bucket_name: str = None
):

    logger.info("Starting data mirroring")

    if not table_names:
        raise ValueError("table_names list cannot be empty")

    base_bucket_path = f"s3a://{bucket_name}/{input_path}"

    # =========================
    # JDBC CONFIG
    # =========================

    jdbc_url = "jdbc:postgresql://localhost:5432/dw"

    connection_properties = {
        "user": "dw_user",
        "password": "dw_password",
        "driver": "org.postgresql.Driver"
    }

    # =========================
    # READ + WRITE LOOP
    # =========================

    for table_name in table_names:

        try:
            logger.info(f"Reading table {table_name}")

            df = spark.read.parquet(f"{base_bucket_path}/{table_name}")

            logger.info(f"Writing table {table_name}")

            (
                df.repartition(4)
                .write
                .format("jdbc")
                .option("url", jdbc_url)
                .option("dbtable", table_name)
                .option("batchsize", 10000)
                .option("numPartitions", 4)
                .options(**connection_properties)
                .mode("append")
                .save()
            )

            logger.info(f"Table {table_name} successfully written")

        except Exception as e:
            logger.error(f"Error processing table {table_name}: {str(e)}")
            raise

    logger.info("Data mirroring finished successfully")