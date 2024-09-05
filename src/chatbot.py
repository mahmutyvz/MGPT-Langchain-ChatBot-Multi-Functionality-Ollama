from src.chat_history import *
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from langchain_community.callbacks import StreamlitCallbackHandler

class Chatbot:
    """
    A class to implement a chatbot with memory management capabilities in Streamlit.

    This class initializes a chatbot capable of handling conversations with 
    different memory configurations, allowing it to remember past interactions 
    and provide context-aware responses.
    """

    def __init__(self):
        """
        Initializes the Chatbot class.

        This constructor synchronizes the Streamlit session state and configures 
        the language model (LLM) to be used for conversation handling.
        """
        session_state_synchronize()
        self.llm = call_llm_model()

    @st.cache_resource
    def create_converstaion_chain(_self,opt,window_num=None):
        """
        Sets up the conversation chain with a specified memory option.

        This method configures the conversation chain for the chatbot using a selected 
        memory type, either `ConversationBufferMemory` or `ConversationBufferWindowMemory`. 
        It initializes the memory based on the provided options and associates it with the 
        language model (LLM).

        Args:
            opt (str): The type of memory to use, either 'ConversationBufferMemory' or 
                       'ConversationBufferWindowMemory'.
            window_num (int, optional): The number of conversation turns to remember 
                                        (only applicable for `ConversationBufferWindowMemory`).

        Returns:
            ConversationChain: The configured conversation chain with the specified memory.
        """
        if opt == 'ConversationBufferMemory':
            memory = ConversationBufferMemory()
        elif opt == 'ConversationBufferWindowMemory':
            memory = ConversationBufferWindowMemory(k=int(window_num))
        chain = ConversationChain(llm=_self.llm, memory=memory, verbose=False)
        return chain

    @clean_chat_history
    def main(self,memory,window_num=None):
        """
        Main function to handle user interactions and conversation execution.

        This method sets up the conversation chain using the provided memory type, captures 
        user input, and displays chat messages. It utilizes the conversation chain to generate 
        context-aware responses based on previous interactions and presents them in the Streamlit 
        chat interface, maintaining chat history.

        Args:
            memory (str): The type of memory to use for the conversation.
            window_num (int, optional): The number of conversation turns to remember (applicable 
                                        for `ConversationBufferWindowMemory`).
        """
        chain = self.create_converstaion_chain(memory,window_num)
        user_query = st.chat_input(placeholder="Ask me anything!")
        if user_query:
            st.session_state.messages.append({"role": "user", "content": user_query})
            show_message(user_query, 'user')
            with st.chat_message("assistant"):
                st_cb = StreamlitCallbackHandler(st.container())
                result = chain.invoke(
                    {"input": user_query},
                    {"callbacks": [st_cb]}
                )
                response = result["response"]
                st.session_state.messages.append({"role": "assistant", "content": response})

