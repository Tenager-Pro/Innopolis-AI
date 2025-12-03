import pandas as pd
from sqlalchemy import text
from pathlib import Path

FILE_PATH = Path(__file__).parent / "customer_churn.csv"

def load_data_from_csv():
    """Load data from CSV file"""
    try:
        df = pd.read_csv(FILE_PATH)
        print("✓ Данные загружены из CSV файла")
        return df
    except FileNotFoundError:
        print("⚠ CSV файл не найден, создаются примеры данных")
        return False
