import streamlit as st
import warnings
import datetime
from src.internet import InternetSearchAccess
from src.chatbot import Chatbot
from src.document import Document
from src.sql import SQL
from src.access import WebAccess

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Langchain",
                   page_icon="🤖", layout="wide")
st.markdown("<h1 style='text-align:center;'>Langchain</h1>", unsafe_allow_html=True)
st.write(datetime.datetime.now(tz=None))

if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]

 
tabs = ["Chatbot","Internet", "Document","Access","SQL","About"]
page = st.sidebar.radio("Tabs", tabs)
if __name__ == "__main__":
    if page == 'Chatbot' or page == 'Access' or page == 'Document':
        memory = st.selectbox(
                "Memory",
                ("ConversationBufferMemory", "ConversationBufferWindowMemory"),
            )
        st.session_state.memory_option = memory
        if memory == 'ConversationBufferWindowMemory':
            window_num = st.number_input("Insert a number", value=5, placeholder="Type a number...")
        else:
            window_num = None
        if page == 'Chatbot':
            chat_obj = Chatbot()
        elif page == 'Document':
            chat_obj = Document()
        elif page == 'Access':
            chat_obj = WebAccess()
        chat_obj.main(memory,window_num)
    else:
        if page == 'Internet': 
            chat_obj = InternetSearchAccess()
            chat_obj.main()
        elif page == 'SQL':
            chat_obj = SQL()
            chat_obj.main()
        elif page == "About":
            st.header("Contact Info")
            st.markdown("""**mahmutyvz324@gmail.com**""")
            st.markdown("""**[LinkedIn](https://www.linkedin.com/in/mahmut-yavuz-687742168/)**""")
            st.markdown("""**[Github](https://github.com/mahmutyvz)**""")
            st.markdown("""**[Kaggle](https://www.kaggle.com/mahmutyavuz)**""")