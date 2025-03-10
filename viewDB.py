import sqlite3
from prettytable import from_db_cursor

def get_all_tables():
    """Retrieve all table names from the database."""
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    
    c.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [table[0] for table in c.fetchall()]  # Extract table names
    
    conn.close()
    return tables

def view_all_tables():
    """Display all tables with their data in a tabular format."""
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    
    tables = get_all_tables()
    
    for table in tables:
        print(f"Table: {table}\n")
        
        # Fetch and display data from each table
        c.execute(f"SELECT * FROM {table}")
        table_data = from_db_cursor(c)
        print(table_data)
    
    conn.close()

# Run the function to display all tables
view_all_tables()
