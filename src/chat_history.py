import streamlit as st
from langchain_community.chat_models import ChatOllama

def clean_chat_history(func):
    """
    Decorator function to manage and enable chat history in the Streamlit app.

    This function decorates another function to maintain the chat session across 
    different pages within the app. It checks the current page and resets the 
    session state (including clearing cached resources) if the user navigates to 
    a different page. It also displays stored messages from the session state 
    to maintain chat continuity.

    Args:
        func (callable): The function to be decorated.

    Returns:
        callable: The wrapped function with enabled chat history.
    """
    def process(*args, **kwargs):
        current_page = func.__qualname__
        if "current_page" not in st.session_state:
            st.session_state["current_page"] = current_page
        if st.session_state["current_page"] != current_page:
            try:
                st.cache_resource.clear()
                del st.session_state["current_page"]
                del st.session_state["messages"]
                st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
            except Exception as e:
                print(f"Error clearing session: {e}")
                
        for msg in st.session_state["messages"]:
            st.chat_message(msg["role"]).write(msg["content"])

        return func(*args, **kwargs)
    return process

def call_llm_model():
    """
    Configures the language model to be used for the chatbot.

    This function initializes and returns an instance of the ChatOllama model 
    from the LangChain community, specifying the model version to be used.

    Returns:
        ChatOllama: An instance of the ChatOllama model configured with the specified version.
    """
    llm = ChatOllama(model="llama3.1")
    return llm

def session_state_synchronize():
    """
    Synchronizes the current Streamlit session state with the app.

    This function iterates over the current session state and updates 
    the Streamlit session state dictionary to ensure all variables and 
    their values are properly synchronized and maintained during the session.
    """
    for k, v in st.session_state.items():
        st.session_state[k] = v

def show_message(msg, author):
    """
    Displays a message in the Streamlit chat interface.

    This function takes a message and an author, and writes the message 
    to the Streamlit chat interface under the specified author's name.

    Args:
        msg (str): The message content to display.
        author (str): The role or name of the author (e.g., "assistant" or "user").
    """
    st.chat_message(author).write(msg)
