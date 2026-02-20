from typing import Callable
from pyspark.sql import SparkSession
import logging

from src.utils.env import load_env_vars

def pyspark_function_wrapper(
    app_name: str,
    pyspark_function: Callable[..., None],
    logger: logging.Logger,
    input_path: str=None,
    output_path: str=None,
    read_from_bucket: bool=False,
    save_to_bucket: bool=False,
    bucket_name: str=None
) -> None:
    logger.info("Creating SparkSession")

    if save_to_bucket or read_from_bucket:
        access_key, secret_key, bucket_name, bucket_endpoint = load_env_vars()

        spark = (
            SparkSession.builder
            .appName(app_name)
            .config(
                "spark.jars.packages",
                "org.apache.hadoop:hadoop-aws:3.4.1,com.amazonaws:aws-java-sdk-bundle:1.12.698,org.postgresql:postgresql:42.7.3"
            )
            .config("spark.hadoop.fs.s3a.endpoint", bucket_endpoint)
            .config("spark.hadoop.fs.s3a.access.key", access_key)
            .config("spark.hadoop.fs.s3a.secret.key", secret_key)
            .config("spark.hadoop.fs.s3a.path.style.access", "true")
            .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
            .getOrCreate()
        )
    else:
        spark = SparkSession.builder.appName(app_name).getOrCreate()

    try:
        pyspark_function(
            spark=spark,
            input_path=input_path,
            output_path=output_path,
            logger=logger,
            read_from_bucket=read_from_bucket,
            save_to_bucket=save_to_bucket,
            bucket_name=bucket_name
        )
    except Exception:
        logger.exception("Error running PySpark job")
        raise
    finally:
        spark.stop()
        logger.info("SparkSession stopped")
