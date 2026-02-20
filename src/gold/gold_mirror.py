from src.utils.wrappers import pyspark_function_wrapper
from src.utils.spark.data_mirroring import data_mirroring
from src.config.logger import logger

if __name__ == "__main__":
    pyspark_function_wrapper(
        app_name="Gold Layer Data Mirroring",
        pyspark_function=data_mirroring,
        logger=logger,
        input_path="gold",
        table_names=[
            "fact_order",
            "dim_date",
            "dim_product"
        ],
        read_from_bucket=True
    )