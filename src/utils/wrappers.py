from typing import Callable
from pyspark.sql import SparkSession
import logging

def pyspark_function_wrapper(
    input_path: str,
    output_path: str,
    app_name: str,
    pyspark_function: Callable[..., None],
    logger: logging.Logger
) -> None:
    logger.info("Creating SparkSession")
    spark = SparkSession.builder.appName(app_name).getOrCreate()

    try:
        pyspark_function(
            spark=spark,
            input_path=input_path,
            output_path=output_path,
            logger=logger
        )
    except Exception:
        logger.exception("Error running PySpark job")
        raise
    finally:
        spark.stop()
        logger.info("SparkSession stopped")
