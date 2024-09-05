import streamlit as st
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
import os
import validators
import traceback
import requests
from langchain_core.documents.base import Document
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings
from src.chat_history import *
from langchain_community.callbacks import StreamlitCallbackHandler

class WebAccess:
    """
    A class for implementing a web-based question-answering (QA) system in Streamlit.

    This class allows users to input URLs, scrape the content of web pages, store them 
    in a vector database, and retrieve answers to user queries based on the scraped content.
    """
    def __init__(self):
        """
        Initializes the WebAccess class.

        This constructor synchronizes the Streamlit session state and configures 
        the language model (LLM) to be used for web-based question-answering.
        """
        session_state_synchronize()
        self.llm = call_llm_model()

    def scrape_website(self,url):
        """
        Scrapes the content of a given website URL.

        This method constructs a request to the website, retrieves its HTML content, and returns 
        the scraped text. If an error occurs during the request, it prints the error traceback.

        Args:
            url (str): The URL of the website to scrape.

        Returns:
            str: The scraped content of the website.
        """
        content = ""
        try:
            base_url = "https://r.jina.ai/"
            final_url = base_url + url
            headers = {
                'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:88.0) Gecko/20100101 Firefox/88.0'
                }
            response = requests.get(final_url, headers=headers)
            content = response.text
        except Exception as e:
            traceback.print_exc()
        return content

    @st.cache_resource(show_spinner='Analyzing webpage', ttl=3600)
    def setup_vectordb(_self,websites):
        """
        Sets up the vector database by scraping and loading content from the provided websites.

        This method scrapes content from each URL, splits the text into chunks, and stores them 
        in a vector database for efficient retrieval.

        Args:
            websites (list): A list of website URLs to scrape and analyze.

        Returns:
            DocArrayInMemorySearch: The vector database containing the split documents.
        """
        docs = []
        for url in websites:
            docs.append(Document(
                page_content=_self.scrape_website(url),
                metadata={"source":url}
                )
            )

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)
        embedding_model = GPT4AllEmbeddings()
        vectordb = DocArrayInMemorySearch.from_documents(splits,embedding_model)
        return vectordb
    
    @st.cache_resource
    def create_cr_chain(_self,_vectordb,opt,window_num=None):
        """
        Sets up the question-answering (QA) chain with a document retriever.

        This method configures a retriever for searching the vector database and initializes 
        a conversational retrieval chain using the selected memory option to handle user queries.

        Args:
            vectordb (DocArrayInMemorySearch): The vector database containing the split documents.
            opt (str): The type of memory to use, either 'ConversationBufferMemory' or 
                       'ConversationBufferWindowMemory'.
            window_num (int, optional): The number of conversation turns to remember 
                                        (only applicable for 'ConversationBufferWindowMemory').

        Returns:
            ConversationalRetrievalChain: The configured conversational retrieval chain.
        """
        retriever = _vectordb.as_retriever(
            search_type='mmr',
            search_kwargs={'k':2, 'fetch_k':4}
        )
        if opt == 'ConversationBufferMemory':
            memory = ConversationBufferMemory(memory_key='chat_history',output_key='answer',return_messages=True)
        elif opt == 'ConversationBufferWindowMemory':
            memory = ConversationBufferWindowMemory(k=int(window_num),memory_key='chat_history',output_key='answer',return_messages=True)
        
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=_self.llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            verbose=False
        )
        return qa_chain

    @clean_chat_history
    def main(self,opt,window_num=None):
        """
        Main function to handle user interactions and web-based question-answering.

        This method allows the user to input website URLs, scrapes the content, stores 
        it in a vector database, and then processes user queries based on the scraped content. 
        It displays both the answers and relevant references in the Streamlit chat interface.

        Args:
            opt (str): The type of memory to use for conversation management.
            window_num (int, optional): The number of conversation turns to remember 
                                        (only applicable for 'ConversationBufferWindowMemory').
        """
        if "websites" not in st.session_state:
            st.session_state["websites"] = []

        web_url = st.sidebar.text_area(
            label='Enter Website URL',
            placeholder="https://",
            help="To add another website, modify this field after adding the website."
            )
        if st.sidebar.button(":heavy_plus_sign: Add Website"):
            valid_url = web_url.startswith('http') and validators.url(web_url)
            if not valid_url :
                st.sidebar.error("Invalid URL! Please check website url that you have entered.", icon="⚠️")
            else:
                st.session_state["websites"].append(web_url)

        if st.sidebar.button("Clear", type="primary"):
            st.session_state["websites"] = []
        
        websites = list(set(st.session_state["websites"]))

        if not websites:
            st.error("Please enter website url to continue!")
            st.stop()
        else:
            st.sidebar.info("Websites - \n - {}".format('\n - '.join(websites)))

            vectordb = self.setup_vectordb(websites)
            qa_chain = self.create_cr_chain(vectordb,opt,window_num)

            user_query = st.chat_input(placeholder="Ask me anything!")
            if websites and user_query:
                st.session_state.messages.append({"role": "user", "content": user_query})
                show_message(user_query, 'user')

                with st.chat_message("assistant"):
                    st_cb = StreamlitCallbackHandler(st.container())
                    result = qa_chain.invoke(
                        {"question":user_query},
                        {"callbacks": [st_cb]}
                    )
                    
                    response = result["answer"]
                    st.session_state.messages.append({"role": "assistant", "content": response})

                    for idx, doc in enumerate(result['source_documents'],1):
                        url = os.path.basename(doc.metadata['source'])
                        ref_title = f":blue[Reference {idx}: *{url}*]"
                        with st.popover(ref_title):
                            st.caption(doc.page_content)