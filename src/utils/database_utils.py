from sqlalchemy import text


def execute_sql_file(engine, file_path):
    """Execute SQL file"""
    with open(file_path, 'r') as file:
        sql_script = file.read()
    
    with engine.connect() as conn:
        # Split by semicolon if multiple statements
        statements = sql_script.split(';')
        for statement in statements:
            if statement.strip():
                conn.execute(text(statement))
        conn.commit()
