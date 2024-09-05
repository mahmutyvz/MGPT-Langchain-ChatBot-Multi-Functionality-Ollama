import streamlit as st
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate
from src.chat_history import *
from langchain_community.callbacks import StreamlitCallbackHandler

class InternetSearchAccess:
    """
    A class to provide internet search access using a chatbot integrated with Streamlit.

    This class initializes a chatbot capable of searching the internet to answer questions
    using DuckDuckGo. It configures the chatbot's language model, sets up the agent with the 
    necessary tools, and manages chat history.
    """
    def __init__(self):
        """
        Initializes the InternetSearchAccess class.

        This constructor synchronizes the Streamlit session state and configures the 
        language model (LLM) to be used for answering queries.
        """
        session_state_synchronize()
        self.llm = call_llm_model()

    def create_agent(_self):
        """
        Sets up the chatbot agent with internet search capabilities.

        This method configures the DuckDuckGo search tool and defines a prompt template 
        for the chatbot. It creates a React agent that uses the configured language model 
        (LLM) and tools to answer user questions. An AgentExecutor is also created to 
        execute the agent's actions.

        Returns:
            AgentExecutor: The configured agent executor for handling user queries.
        """
        ddg_search = DuckDuckGoSearchRun()
        tools = [
            Tool(
                name="DuckDuckGoSearch",
                func=ddg_search.run,
                description="Useful for when you need to answer questions about current events. You should ask targeted questions",
            )
        ]       

        template = '''You are a helpful assistant who can answer questions using a set of tools. Answer the following questions to the best of your ability. You have access to the following tools:

        {tools}

        When answering, please use the following format:

        Question: The input question you must answer
        Thought: Describe your thought process and reasoning here
        Action: The action to take, should be one of [{tool_names}]
        Action Input: The input to the action
        Observation: The result of the action
        Thought: Reflect on the observation and continue
        Final Answer: The final answer to the original input question

        Here is the question:

        Question: {input}
        {agent_scratchpad}'''


        prompt = PromptTemplate.from_template(template)
    
        agent = create_react_agent(_self.llm, tools, prompt)
        agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools,  
        verbose=True, 
        max_iterations=20,  
        handle_parsing_errors=True
        )

        return agent_executor

    @clean_chat_history
    def main(self):
        """
        Main function to handle user interactions and query execution.

        This method sets up the agent, captures user input, and displays chat messages.
        It uses the configured agent executor to handle user queries and provides responses 
        through the Streamlit chat interface, maintaining chat history.

        If a user query is entered, the function adds it to the chat history, executes the 
        agent to get the response, and then displays the response in the chat interface.
        """
        agent_executor=self.create_agent()
        user_query = st.chat_input(placeholder="Ask me anything!")
        if user_query:
            st.session_state.messages.append({"role": "user", "content": user_query})
            show_message(user_query, 'user')
            with st.chat_message("assistant"):
                st_cb = StreamlitCallbackHandler(st.container())
                result = agent_executor.invoke(
                    {"input": user_query,},
                    callbacks=[st_cb]
                )
                response = result["output"]
                
                st.session_state.messages.append({"role": "assistant", "content": response})