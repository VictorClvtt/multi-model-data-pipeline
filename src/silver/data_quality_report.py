import warnings
import pandas as pd
import great_expectations as ge
import glob

def data_quality_report(
        dataset_path,
        dataset="raw"
    ):

    warnings.filterwarnings("ignore", category=UserWarning)

    if dataset.lower() == "raw":
        df = pd.read_csv(
            dataset_path,
            parse_dates=["order_date", "shipping_date"]
        )
    elif dataset.lower() == "clean":
        # Lista todos os arquivos CSV dentro da pasta
        csv_files = glob.glob(dataset_path)

        # Lê todos e concatena em um DataFrame
        df = pd.concat(
            [pd.read_csv(f, parse_dates=["order_date", "shipping_date"]) for f in csv_files],
            ignore_index=True
        )

    df["_shipping_after_order"] = (
        df["shipping_date"].notna() &
        (df["shipping_date"] > df["order_date"])
    )

    ge_df = ge.from_pandas(df)

    VALID_STATUS = ["Entregue", "Em trânsito", "Cancelado"]
    VALID_PAYMENT = ["Cartão de crédito", "Boleto", "Pix"]
    VALID_STATES = ["SP", "RJ", "MG", "PR", "RS", "SC", "BA", "PE", "CE", "GO", "DF"]

    CANCELLED_CONDITION = 'status == "Cancelado"'

    # ------------------------------------------------------------------
    # CUSTOMER
    # ------------------------------------------------------------------

    # Null checks
    ge_df.expect_column_values_to_not_be_null("customer_id")
    ge_df.expect_column_values_to_not_be_null("customer_name")
    ge_df.expect_column_values_to_not_be_null("customer_email")

    # String hygiene
    ge_df.expect_column_values_to_not_match_regex(
        "customer_name",
        r"^\s|\s$"
    )

    # Email format
    ge_df.expect_column_values_to_match_regex(
        "customer_email",
        r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
    )

    # Domain
    ge_df.expect_column_values_to_be_in_set(
        "customer_state",
        VALID_STATES
    )

    # ------------------------------------------------------------------
    # ORDER
    # ------------------------------------------------------------------

    # Duplicates and nulls
    ge_df.expect_column_values_to_be_unique("order_id")
    ge_df.expect_column_values_to_not_be_null("order_id")
    ge_df.expect_column_values_to_not_be_null("order_date")

    # Domain
    ge_df.expect_column_values_to_be_in_set("status", VALID_STATUS)

    # Relationship
    ge_df.expect_column_values_to_not_be_null("customer_id")

    # ------------------------------------------------------------------
    # ORDER_ITEM
    # ------------------------------------------------------------------

    # Null checks
    ge_df.expect_column_values_to_not_be_null("product_id")
    ge_df.expect_column_values_to_not_be_null("order_id")
    ge_df.expect_column_values_to_not_be_null("quantity")
    ge_df.expect_column_values_to_not_be_null("unit_price")

    # Logical ranges
    ge_df.expect_column_values_to_be_between(
        "quantity",
        min_value=1
    )

    ge_df.expect_column_values_to_be_between(
        "unit_price",
        min_value=0.01,
        mostly=0.95
    )

    # ------------------------------------------------------------------
    # PAYMENT
    # ------------------------------------------------------------------

    ge_df.expect_column_values_to_be_in_set(
        "payment_method",
        VALID_PAYMENT
    )

    ge_df.expect_column_values_to_not_be_null("order_id")

    # ------------------------------------------------------------------
    # SHIPPING
    # ------------------------------------------------------------------

    ge_df.expect_column_values_to_be_null(
        "shipping_date",
        row_condition=CANCELLED_CONDITION,
        condition_parser="pandas"
    )

    ge_df.expect_column_values_to_be_in_set(
        "_shipping_after_order",
        [True],
        mostly=0.98,
        row_condition="shipping_date.notnull()",
        condition_parser="pandas"
    )

    # ------------------------------------------------------------------
    # REVIEW
    # ------------------------------------------------------------------

    ge_df.expect_column_values_to_be_null(
        "review",
        row_condition=CANCELLED_CONDITION,
        condition_parser="pandas"
    )

    # ------------------------------------------------------------------
    # Validação
    # ------------------------------------------------------------------
    results = ge_df.validate()

    # ------------------------------------------------------------------
    # Relatório
    # ------------------------------------------------------------------
    print("Success:", results["success"])
    print("Total expectations:", len(results["results"]))

    failed = [r for r in results["results"] if not r["success"]]
    print("Failed expectations:", len(failed))

    for r in failed:
        exp_type = r["expectation_config"]["expectation_type"]
        kwargs = r["expectation_config"]["kwargs"]
        column = kwargs.get("column", kwargs.get("column_A", "N/A"))
        print(f"[FAIL] {exp_type} | column: {column}")

    print("[INFO] Data Quality report generated successfully.")

if __name__ == "__main__":
    data_quality_report(dataset="raw", dataset_path="./data/bronze/order_data.csv")
    # data_quality_report(dataset="clean", dataset_path="./data/clean/clean_orders/*.csv")