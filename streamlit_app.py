import pandas as pd
import streamlit as st
from st_keyup import st_keyup
from langchain_community.utilities import SQLDatabase
import psycopg2
from psycopg2 import Error

def init_database() -> SQLDatabase:
    try:
        # PostgreSQL connection string
        db_uri = f"postgresql+psycopg2://YZyadat.py:Yhyayhyayhya1@localhost:5432/chinook"
        return SQLDatabase.from_uri(db_uri)
    except Error as e:
        st.error(f"Error connecting to PostgreSQL: {e}")
        return None

@st.cache_data
def get_projects() -> pd.DataFrame:
    try:
        # Direct PostgreSQL connection for pandas
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            user="YZyadat.py",
            password="Yhyayhyayhya1",
            database="chinook"
        )
        
        query = "SELECT * FROM project"  # Your table name
        df = pd.read_sql(query, conn)
        return df
    except Error as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()
    finally:
        if 'conn' in locals():
            conn.close()

projects = get_projects()

debounce = st.checkbox("Add 0.5s debounce?")

disabled = st.checkbox("Disable input?")

name = st_keyup(
    "Enter project name", debounce=500 if debounce else None, disabled=disabled
)

if name:
    filtered = projects[projects.name.str.lower().str.contains(name.lower(), na=False)]
else:
    filtered = projects

st.write(len(filtered), "projects found")
st.write(filtered)
