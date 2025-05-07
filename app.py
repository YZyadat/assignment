import streamlit as st
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_groq import ChatGroq
import pandas as pd
import streamlit as st
from st_keyup import st_keyup
from langchain_community.utilities import SQLDatabase
import psycopg2
from psycopg2 import Error
load_dotenv()

def init_database(user: str, password: str, host: str, port: str, database: str) -> SQLDatabase:
#   db_uri = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
  db_uri = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
  return SQLDatabase.from_uri(db_uri)

def get_sql_chain(db):
    template = """
        You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
        Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account. make sure the sql query has no limits in number in order to show all the related tables records.
        
        <SCHEMA>{schema}</SCHEMA>
        
        Conversation History: {chat_history}
        
        Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.
        
        
        Question: {question}
        For example:
            Question: which 3 artists have the most tracks?
            SQL Query: SELECT ArtistId, COUNT(*) as track_count FROM Track GROUP BY ArtistId ORDER BY track_count DESC;
            Question: Name 10 artists
            SQL Query: SELECT Name FROM Artist;
        SQL Query:
    """

    def get_schema(_):
        return db.get_table_info()
    
    prompt = ChatPromptTemplate.from_template(template)

    # llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)
    llm = ChatOpenAI(model="gpt-4-0125-preview")

    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )
    
def get_reponse(user_query: str, chat_history: list, db: SQLDatabase,summarize_enabled: bool = False, comparison_enabled: bool = False, projects_end_date_table: bool = False):
    sql_chain = get_sql_chain(db)

    summarize_instruction = "And provide a summary of the asked project." if summarize_enabled else ""
    comparison_instruction = "And in the end add comparison table that compares the asked project with other projects based on cost, start date, end date, and status." if comparison_enabled else ""
    projects_end_date_table_instruction = "And please include table that shows the projects timeline." if projects_end_date_table else ""

    template = """
        Given the database with the following tables: <SCHEMA>{schema}</SCHEMA>
        When I request a summary or detailed report about a specific project, use all related data across the givin tables. The response should include:
        Basic project information from the Projects table.
        All related Tasks with assigned Employees and their roles.
        The Departments responsible for or collaborating on the project (via Involved Departments).
        A list of beneficiaries (departments or employees) from the project.
        Any interdependencies or patterns that can be inferred (e.g., cross-department collaboration, employee involvement across multiple projects).
        Respond in a concise, clear, and structured format.

        {summarize_instruction}

        IMPORTANT: Make sure to include ALL projects in the analysis. If a specific project is mentioned, provide detailed information about that project, but still include it in the overall comparison.

        {comparison_instruction}

        {projects_end_date_table_instruction}

        Make sure the response is generated for someone who is not technical. So find proper terminologies and avoid technical words such as table, id, etc..

        Conversation History: {chat_history}
        SQL Query: <SQL>{query}</SQL>
        User question: {question}
        SQL Response: {response}
    """


    prompt = ChatPromptTemplate.from_template(template)

    # llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)
    llm = ChatOpenAI(temperature=0.7)
    full_chain = (
        RunnablePassthrough
        .assign(query=sql_chain)
        .assign(schema=lambda _: db.get_table_info(),
                response=lambda vars: db.run(vars["query"]))
                
        | prompt
        | llm
        | StrOutputParser()
    )

    data = {
        "question": user_query,
        "chat_history": chat_history,
        "summarize_instruction": summarize_instruction,
        "comparison_instruction": comparison_instruction,
        "projects_end_date_table_instruction": projects_end_date_table_instruction
        }

    return full_chain.invoke(data)

@st.cache_data
def get_projects() -> pd.DataFrame:
    try:
        conn = psycopg2.connect(
            host=st.session_state["Host"],
            port=st.session_state["Port"],
            user=st.session_state["User"],
            password=st.session_state["Password"],
            database=st.session_state["Database"]
        )
        
        query = "SELECT * FROM project"
        df = pd.read_sql(query, conn)
        
        return df
    except Error as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()
    finally:
        if 'conn' in locals():
            conn.close()


if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
      AIMessage(content="Hello! I'm a SQL assistant. Ask me anything about your database."),
    ]

st.set_page_config(page_title="Projects Analysis", page_icon=":speech_balloon:")
st.title("Projects Analysis")


if "connected" not in st.session_state:
    st.session_state.connected = False

with st.sidebar:
    st.subheader("Settings")
    st.write("This is a simple chat application using MySQL. Connect to the database and start chatting.")
    
    st.text_input("Host", value="localhost", key="Host")
    st.text_input("Port", value="5432", key="Port")
    st.text_input("User", value="YZyadat.py", key="User")
    st.text_input("Password", type="password", value="Yhyayhyayhya1", key="Password")
    st.text_input("Database", value="chinook", key="Database")

    if st.button("Connect"):
        with st.spinner("Connecting to database..."):
            db = init_database(
                st.session_state["User"],
                st.session_state["Password"],
                st.session_state["Host"],
                st.session_state["Port"],
                st.session_state["Database"]
            )
            st.session_state.db = db
            st.session_state.connected = True
            st.success("Connected to database!")


if st.session_state.connected:
    projects = get_projects()
    
    summarize_enabled = st.checkbox("Project Summary")
    comparison_enabled = st.checkbox("Comparison With Other Projects")
    projects_end_date_table = st.checkbox("Projects Timeline")

    name = st_keyup(
                "Search by Project Name"
            )
    

    if name:
        filtered = projects[projects.name.str.lower().str.contains(name.lower(), na=False)]
    else:
        filtered = projects

    st.write(len(filtered), "projects found")
    st.write(filtered)
    

if st.session_state.connected:
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.markdown(message.content)

    user_query = st.chat_input("Type a message...")

    if user_query is not None and user_query.strip() != "":
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        
        with st.chat_message("Human"):
            st.markdown(user_query)
            
        with st.chat_message("AI"):
            response = get_reponse(user_query,st.session_state.chat_history,
                st.session_state.db,
                summarize_enabled,
                comparison_enabled,
                projects_end_date_table)
            st.markdown(response)
            
        st.session_state.chat_history.append(AIMessage(content=response))
