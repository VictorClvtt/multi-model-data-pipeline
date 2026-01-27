# %%
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.window import Window
import logging

# Logger configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

INPUT_PATH = "./data/bronze/order_data.csv"
OUTPUT_PATH = "./data/silver/clean_orders"

spark = SparkSession.builder.appName("Clean Raw Data").getOrCreate()

logger.info("Starting raw data ingestion")
df = spark.read.csv(
    INPUT_PATH,
    header=True,
    inferSchema=True
)
logger.info("Raw data successfully loaded")

# %%
# Removendo linhas duplicadas
initial_count = df.count()
logger.info("Removing duplicate rows based on order_id")
df = df.drop_duplicates(subset=["order_id"])
logger.info(
    f"Duplicate removal completed. Rows before: {initial_count}, after: {df.count()}"
)

# %%
# Removendo espaços em branco antes e/ou depois de nomes
logger.info("Trimming leading and trailing whitespaces from customer_name")
df = df.withColumn(
    "customer_name",
    F.trim(F.col("customer_name"))
)

# %%
# Imputando valores de preço pela média de cada produto sobre valores nulos(valor padrão de cada produto)
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
logger.info(f"Remaining null unit_price values after imputation: {remaining_null_prices}")

# %%
# Removendo linhas com valores de quantidade fora da faixa de valores lógica
logger.info("Removing rows with invalid quantity values (quantity < 1)")

df = df.where(
    F.col("quantity") >= 0,
)

logger.info("Invalid quantity rows removed")

# %%
# Tratando datas invertidas assumindo erro do sistema de origem (swap entre order_date e shipping_date)
logger.info(
    "Fixing inverted dates assuming a source system error (swap between order_date and shipping_date)"
)

df = df.dropna(subset=["order_date"], how="any")

df = df.withColumn("order_date_tmp", F.col("order_date"))

# Ajusta order_date apenas se shipping_date não for nulo
df = df.withColumn(
    "order_date",
    F.when(
        (F.col("shipping_date").isNotNull()) & (F.col("order_date_tmp") > F.col("shipping_date")),
        F.col("shipping_date")
    ).otherwise(F.col("order_date_tmp"))
)

# Ajusta shipping_date apenas se não for nulo
df = df.withColumn(
    "shipping_date",
    F.when(
        (F.col("shipping_date").isNotNull()) & (F.col("order_date_tmp") > F.col("shipping_date")),
        F.col("order_date_tmp")
    ).otherwise(F.col("shipping_date"))
).drop("order_date_tmp")

logger.info("Date correction completed")

# %%
# Padronizando e-emails

# 1. Limpeza básica
df = df.withColumn("customer_email", F.trim(F.col("customer_email")))
df = df.withColumn("customer_email", F.lower(F.col("customer_email")))
df = df.withColumn("customer_email", F.regexp_replace(F.col("customer_email"), r"\s+", ""))
df = df.withColumn("customer_email", F.regexp_replace(F.col("customer_email"), r"[^a-z0-9@._-]", ""))
df = df.withColumn("customer_email", F.regexp_replace(F.col("customer_email"), r"@+", "@"))

# 2. Garantir pelo menos um ponto no domínio
df = df.withColumn(
    "customer_email",
    F.when(
        F.col("customer_email").rlike(r"@[^@]+\.[^@]+"),
        F.col("customer_email")
    ).otherwise(None)
)

# 3. Drop linhas com emails que ainda não são válidos
df = df.dropna(subset=["customer_email"])
logger.info("E-mail address correction completed")

# %%
# Escrita do dataset limpo
logger.info("Writing cleaned dataset to output path")
df.write.mode("overwrite").csv(OUTPUT_PATH, header=True)
logger.info(f"Cleaned dataset successfully written, total rows: {df.count()}")
# %%
