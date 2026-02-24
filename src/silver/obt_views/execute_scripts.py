from pathlib import Path

from src.utils.sql import execute_sql_file

def create_obt_views(folder_path):
    sql_folder = folder_path

    print(list(sql_folder.glob("*.sql")))

    for file in sorted(sql_folder.glob("*.sql")):
        execute_sql_file(file)

if __name__ == "__main__":
    create_obt_views(Path(__file__).parent)