# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'postgres', 
    'password': 'root',  # Change to your password
    'port': '5432',
    'database': 'customer_analysis_db'
}

def get_connection_string():
    return f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
