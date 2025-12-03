#!/usr/bin/env python3
import psycopg2
import pandas as pd
from sqlalchemy import create_engine

from src.config.database import DB_CONFIG, get_connection_string
from src.utils.database_utils import execute_sql_file
from data.load_data import load_data_from_csv

def create_database():
    """Create database if it doesn't exist"""
    try:
        # Connect to default postgres database
        conn = psycopg2.connect(
            host=DB_CONFIG['host'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            port=DB_CONFIG['port'],
            database='postgres'
        )
        conn.autocommit = True
        cur = conn.cursor()
        
        # Check if database exists
        cur.execute(f"SELECT 1 FROM pg_catalog.pg_database WHERE datname = '{DB_CONFIG['database']}'")
        exists = cur.fetchone()
        
        if not exists:
            cur.execute(f"CREATE DATABASE {DB_CONFIG['database']}")
            print(f"✓ Database '{DB_CONFIG['database']}' created")
        else:
            print(f"✓ Database '{DB_CONFIG['database']}' already exists")
            
        cur.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"✗ Error creating database: {e}")
        return False

def main():
    """Main execution function"""
    print("🚀 STARTING CUSTOMER ANALYSIS PIPELINE")
    
    # Create database
    if not create_database():
        return
    
    # Create engine
    engine = create_engine(get_connection_string())
    
    try:
        # Delete view if exists
        execute_sql_file(engine, 'sql/drop_layers.sql')
        print("✓ Layers dropped")

        # Execute SQL files
        print("\n📊 SETTING UP DATABASE SCHEMA")
        execute_sql_file(engine, 'sql/schema.sql')
        execute_sql_file(engine, 'sql/queries.sql')
        
        # Create sample data if no CSV exists
        try:
            df = load_data_from_csv()
            print("✓ Data loaded from CSV")
        except:
            print("⚠ No CSV found, creating sample data")
        
        # Save to database
        df.to_sql('raw_layer', engine, if_exists='replace', index=False)
        print("✓ Data saved to database")

        execute_sql_file(engine, 'sql/layers.sql')
        print("✓ Updated layers table")
        
        # Verify setup
        print("\n✅ PIPELINE SETUP COMPLETED SUCCESSFULLY!")
        print("\nNext steps:")
        print("1. Run notebooks/01_data_preparation.ipynb for data processing")
        print("2. Run notebooks/02_baseline_model.ipynb for baseline models") 
        print("3. Run notebooks/03_model_experiments.ipynb for model tuning")
        print("4. Run notebooks/04_model_experiments.ipynb for final model")
        
    except Exception as e:
        print(f"✗ Error in pipeline: {e}")

if __name__ == "__main__":
    main()
