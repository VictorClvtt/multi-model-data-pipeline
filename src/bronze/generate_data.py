import pandas as pd
import numpy as np
from faker import Faker

from src.utils.bucket import save_to_minio

def generate_data(
    seed: int,
    num_orders: int,
    num_customers: int,
    save_to_bucket: bool,
    output_path: str = None
):
    # ------------------------------
    # Configurações das bibliotecas
    # ------------------------------
    Faker.seed(seed)

    fake = Faker("pt_BR")
    np.random.seed(seed)

    # ------------------------
    # Configurações dos dados
    # ------------------------

    PAYMENT_METHODS = ["Cartão de crédito", "Boleto", "Pix"]
    STATUS = ["Entregue", "Em trânsito", "Cancelado"]

    REVIEWS = [
        "Produto chegou antes do esperado",
        "Qualidade ruim",
        "Excelente atendimento",
        "Não recomendo",
        "Muito bom, recomendo",
        "Entrega atrasada"
    ]

    # ---------------------------
    # Catálogo único de produtos
    # ---------------------------
    PRODUCT_CATALOG = {
        "Smartphone Alpha": {"category": "Eletrônicos", "supplier": "TechBrasil", "price": 3299.90, "weight": 0.18},
        "Notebook Beta": {"category": "Eletrônicos", "supplier": "Eletrônicos do Sul", "price": 4599.00, "weight": 0.15},
        "Fone Gamma": {"category": "Eletrônicos", "supplier": "TechBrasil", "price": 499.90, "weight": 0.10},
        "Câmera Delta": {"category": "Eletrônicos", "supplier": "Eletrônicos do Sul", "price": 2899.00, "weight": 0.05},

        "Camiseta Basic": {"category": "Moda", "supplier": "ModaTop", "price": 99.90, "weight": 0.12},
        "Tênis Runner": {"category": "Moda", "supplier": "Estilo Urbano", "price": 399.90, "weight": 0.10},
        "Jaqueta Urban": {"category": "Moda", "supplier": "ModaTop", "price": 599.90, "weight": 0.06},

        "Python para Iniciantes": {"category": "Livros", "supplier": "Livraria Central", "price": 79.90, "weight": 0.04},
        "SQL Avançado": {"category": "Livros", "supplier": "BookStore BR", "price": 119.90, "weight": 0.03},
        "Engenharia de Dados": {"category": "Livros", "supplier": "Livraria Central", "price": 149.90, "weight": 0.03},

        "Mesa Madeira": {"category": "Casa", "supplier": "MoveisCenter", "price": 899.00, "weight": 0.03},
        "Cadeira Confort": {"category": "Casa", "supplier": "Casa & Cia", "price": 499.00, "weight": 0.02},
        "Luminária Zen": {"category": "Casa", "supplier": "Casa & Cia", "price": 199.90, "weight": 0.02},

        "Bola Pro": {"category": "Esporte", "supplier": "SportPro", "price": 149.90, "weight": 0.03},
        "Mochila Trek": {"category": "Esporte", "supplier": "Esporte Mania", "price": 299.90, "weight": 0.03},
        "Luvas Fit": {"category": "Esporte", "supplier": "SportPro", "price": 89.90, "weight": 0.01}
    }

    # -----------------
    # Dimensão produto
    # -----------------
    products = pd.DataFrame([
        {
            "product_id": i + 1,
            "product_name": name,
            "category": attrs["category"],
            "supplier": attrs["supplier"],
            "unit_price": attrs["price"],
            "weight": attrs["weight"]
        }
        for i, (name, attrs) in enumerate(PRODUCT_CATALOG.items())
    ])

    # Normalizar pesos
    products["weight"] = products["weight"] / products["weight"].sum()

    # ---------------------------------------------
    # Distribuição de estados (tendência regional)
    # ---------------------------------------------
    STATE_WEIGHTS = {
        "SP": 0.32, "RJ": 0.15, "MG": 0.13,
        "PR": 0.07, "RS": 0.07, "SC": 0.06,
        "BA": 0.05, "PE": 0.04, "CE": 0.04,
        "GO": 0.04, "DF": 0.03
    }

    states = list(STATE_WEIGHTS.keys())
    state_probs = list(STATE_WEIGHTS.values())

    # ---------
    # Clientes
    # ---------
    customers = pd.DataFrame({
        "customer_id": range(1, num_customers + 1),
        "customer_name": [fake.name() for _ in range(num_customers)],
        "customer_email": [fake.email() for _ in range(num_customers)],
        "customer_state": np.random.choice(states, num_customers, p=state_probs)
    })

    customers["customer_city"] = customers["customer_state"].apply(
        lambda _: fake.city()
    )

    # ---------------------------
    # Pedidos (produto com peso)
    # ---------------------------
    orders = pd.DataFrame({
        "order_id": range(1, num_orders + 1),
        "customer_id": np.random.choice(customers["customer_id"], num_orders),
        "product_id": np.random.choice(
            products["product_id"],
            num_orders,
            p=products["weight"]
        ),
        "quantity": np.random.randint(1, 5, num_orders),
        "payment_method": np.random.choice(PAYMENT_METHODS, num_orders),
        "status": np.random.choice(STATUS, num_orders, p=[0.85, 0.10, 0.05]),
        "order_date": pd.to_datetime(
            np.random.randint(
                pd.Timestamp("2025-01-01").value // 10**9,
                pd.Timestamp("2025-12-31").value // 10**9,
                num_orders
            ),
            unit="s"
        )
    })

    orders["shipping_date"] = orders["order_date"] + pd.to_timedelta(
        np.random.randint(1, 8, num_orders), unit="D"
    )

    orders["review"] = np.random.choice(REVIEWS, num_orders)

    # --------------
    # Combinar tudo
    # --------------
    data = (
        orders
        .merge(customers, on="customer_id")
        .merge(products.drop(columns="weight"), on="product_id")
    )

    # ------------------
    # Defeitos de dados
    # ------------------
    data.loc[data.sample(frac=0.02).index, "review"] = None
    data.loc[data.sample(frac=0.05).index, "unit_price"] = None
    data.loc[data.sample(frac=0.005).index, "quantity"] = -1

    idx = data.sample(frac=0.002).index
    data.loc[idx, "shipping_date"] = data.loc[idx, "order_date"] - pd.to_timedelta(
        np.random.randint(1, 5, len(idx)), unit="D"
    )

    data = pd.concat([data, data.sample(frac=0.005)], ignore_index=True)

    idx = data.sample(frac=0.01).index
    data.loc[idx, "customer_email"] = data.loc[idx, "customer_email"].str.replace("@", " at ")

    idx = data.sample(frac=0.01).index
    data.loc[idx, "customer_name"] = data.loc[idx, "customer_name"].apply(lambda x: f" {x} ")

    cancelados = data["status"] == "Cancelado"
    data.loc[cancelados, "review"] = None
    data.loc[cancelados, "shipping_date"] = pd.NaT

    # ---------------
    # Colunas finais
    # ---------------
    data = data[[
        "order_id", "order_date", "shipping_date", "status", "quantity",
        "unit_price", "payment_method",
        "customer_id", "customer_name", "customer_email",
        "customer_city", "customer_state",
        "product_id", "product_name", "category", "supplier",
        "review"
    ]]

    # -------
    # Salvar
    # -------
    if save_to_bucket:
        return data
    else:
        data.to_csv(output_path, index=False)

    print(f"[INFO] {num_orders} synthetic order records generated successfully.")

if __name__ == "__main__":
    save_to_minio(
        df=generate_data(
            seed=42,
            num_orders=12500,
            num_customers=1500,
            save_to_bucket=True
        ),
        bucket="orders-data",
        key="bronze/CSV/order_data.csv"
    )