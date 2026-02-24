import psycopg2
from pathlib import Path

def execute_sql_file(filepath: str):

    conn = psycopg2.connect(
        host="localhost",
        database="dw",
        user="dw_user",
        password="dw_password",
        port=5432
    )

    conn.autocommit = True
    cursor = conn.cursor()

    sql = Path(filepath).read_text()

    cursor.execute(sql)

    cursor.close()
    conn.close()

    print(f"{filepath} executed successfully")