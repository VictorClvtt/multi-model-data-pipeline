from src.utils.wrappers import pyspark_function_wrapper
from src.utils.spark.data_mirroring import data_mirroring
from src.config.logger import logger

from datetime import datetime

if __name__ == "__main__":
    pyspark_function_wrapper(
        app_name="Silver Layer OBT Data Mirroring",
        pyspark_function=data_mirroring,
        logger=logger,
        input_path="silver",
        table_names=["obt"],
        read_from_bucket=True
    )