import streamlit as st
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.embeddings import GPT4AllEmbeddings
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.chains.sql_database.prompt import MSSQL_PROMPT,PROMPT_SUFFIX
from langchain.prompts import FewShotPromptTemplate, SemanticSimilarityExampleSelector
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from src.chat_history import *
from paths import Path

class SQL:
    """
    A class for implementing an SQL-based question-answering (QA) system in Streamlit.

    This class allows users to query a SQL database using natural language inputs. 
    It configures an LLM-based SQL agent to interpret and execute SQL queries based on user inputs.
    """
    def __init__(self):
        """
        Initializes the SQL class.

        This constructor synchronizes the Streamlit session state and configures 
        the language model (LLM) to be used for SQL-based question-answering.
        """
        session_state_synchronize()
        self.llm = call_llm_model()
    
    def connect_db(self):
        """
        Sets up the SQL database connection and retrieves table names.

        This method establishes a connection to the SQL database using the connection 
        string defined in `Path.sql_path`. It also retrieves and displays usable table names in the sidebar.

        Returns:
            SQLDatabase: The configured SQLDatabase object for the connected database.
        """
        conn_str = Path.sql_path
        db_engine = create_engine(conn_str)
        db = SQLDatabase(db_engine)
        
        with st.sidebar.expander('Tables', expanded=True):
            st.info('\n- '+'\n- '.join(db.get_usable_table_names()))
        return db
    def create_sql_agent(self,db):
        """
        Sets up the SQL agent with few-shot learning for SQL query generation.

        This method configures the SQL agent using a few-shot prompt template that includes 
        examples of SQL queries. The examples are embedded using GPT-4All embeddings, and a 
        similarity-based example selector is used to find the most relevant examples for the user's query.

        Args:
            db (SQLDatabase): The SQLDatabase object for the connected database.

        Returns:
            SQLDatabaseChain: The configured SQL agent chain for question-answering.
        """
        few_shots = [
            {'Question' : "Can you fetch the TOWNS with CITYID 8 from the TOWNS table?",
            'SQLQuery' : "SELECT * FROM [TOWNS] WHERE [CITYID] = 8",
            'SQLResult': "Result of the SQL query",
            'Answer' : "[(44, 8, 'ARDANUÇ'), (46, 8, 'ARHAVİ'), (51, 8, 'ARTVİN MERKEZ'), (101, 8, 'BORÇKA'), (292, 8, 'HOPA'), (546, 8, 'ŞAVŞAT'), (629, 8, 'YUSUFELİ'), (721, 8, 'MURGUL')]"},
            {'Question' : "Can you bring the 5 highest TOTALPRICE values ​​from the ORDERS table?",
            'SQLQuery' : "SELECT TOP 5 [TOTALPRICE] FROM [ORDERS] ORDER BY [TOTALPRICE] DESC",
            'SQLResult': "Result of the SQL query",
            'Answer' : "[(Decimal('14024.3663'),), (Decimal('12636.1377'),), (Decimal('12375.1248'),), (Decimal('12310.9373'),), (Decimal('11905.6570'),)]"},
        ]
        embeddings = GPT4AllEmbeddings()
        to_vectorize = [" ".join(example.values()) for example in few_shots]
        vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=few_shots)
        example_selector = SemanticSimilarityExampleSelector(
        vectorstore=vectorstore,
        k=2,
        )
        example_prompt = PromptTemplate(
            input_variables=["Question", "SQLQuery", "SQLResult","Answer",],
            template="\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer: {Answer}",
        )
        few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix=MSSQL_PROMPT.template,
        suffix=PROMPT_SUFFIX,
        input_variables=["input", "table_info", "top_k"],
        )
        db_chain=SQLDatabaseChain.from_llm(self.llm,db,verbose=True,prompt=few_shot_prompt)
        return db_chain

    @clean_chat_history
    def main(self):
        """
        Main function to handle user interactions and SQL-based question-answering.

        This method sets up the SQL database and agent, takes user input as a natural language query, 
        and processes it using the SQL agent to retrieve and display the results in the Streamlit chat interface.
        """
        db = self.connect_db()
        agent = self.create_sql_agent(db)

        user_query = st.chat_input(placeholder="Ask me anything!")

        if user_query:
            st.session_state.messages.append({"role": "user", "content": user_query})
            show_message(user_query, 'user')

            with st.chat_message("assistant"):
                st_cb = StreamlitCallbackHandler(st.container())
                result = agent.invoke(
                    {"query": user_query},
                    {"callbacks": [st_cb]}
                )
                
                response = result['result']
                st.session_state.messages.append({"role": "assistant", "content": response})