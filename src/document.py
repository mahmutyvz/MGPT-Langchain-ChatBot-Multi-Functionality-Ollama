import streamlit as st
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
import os
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings
from src.chat_history import *
from langchain_community.callbacks import StreamlitCallbackHandler
from paths import Path

class Document:
    """
    A class to implement a document-based question-answering (QA) system in Streamlit.

    This class provides functionalities to upload PDF documents, analyze their content,
    and answer user queries based on the content of these documents using a conversational
    retrieval chain.
    """
    def __init__(self):
        """
        Initializes the Document class.

        This constructor synchronizes the Streamlit session state and configures 
        the language model (LLM) to be used for document-based question-answering.
        """
        session_state_synchronize()
        self.llm = call_llm_model()   

    def save_file(self,file):
        """
        Saves an uploaded file to the specified directory.

        This method checks if the folder exists, creates it if necessary, and saves 
        the uploaded file in that folder.

        Args:
            file (UploadedFile): The uploaded PDF file from the user.

        Returns:
            str: The path to the saved file.
        """
        folder = Path.pdf_save_path
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        file_path = f'./{folder}/{file.name}'
        with open(file_path, 'wb') as f:
            f.write(file.getvalue())
        return file_path
    
    @st.cache_resource
    @st.spinner('Analyzing documents..')
    def create_cr_chain(_self,uploaded_files,opt,window_num=None):
        """
        Sets up the question-answering chain with document retrieval capabilities.

        This method loads the uploaded documents, splits them into chunks, stores them
        in a vector database, and configures a retriever for searching relevant content.
        It also initializes a conversational retrieval chain using the selected memory 
        option to handle user queries.

        Args:
            uploaded_files (list): A list of uploaded PDF files from the user.
            opt (str): The type of memory to use, either 'ConversationBufferMemory' or 
                       'ConversationBufferWindowMemory'.
            window_num (int, optional): The number of conversation turns to remember 
                                        (only applicable for 'ConversationBufferWindowMemory').

        Returns:
            ConversationalRetrievalChain: The configured conversational retrieval chain.
        """
        docs = []
        for file in uploaded_files:
            file_path = _self.save_file(file)
            loader = PyPDFLoader(file_path)
            docs.extend(loader.load())
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)
        embedding_model = GPT4AllEmbeddings()
        vectordb = DocArrayInMemorySearch.from_documents(splits, embedding_model)

        retriever = vectordb.as_retriever(
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
    def main(self,memory,window_num=None):
        """
        Main function to handle user interactions and document-based question-answering.

        This method allows the user to upload PDF documents and enter queries. It uses the 
        configured QA chain to retrieve answers from the documents' content and displays 
        both the answer and relevant references in the Streamlit chat interface.

        Args:
            memory (str): The type of memory to use for conversation management.
            window_num (int, optional): The number of conversation turns to remember 
                                        (only applicable for 'ConversationBufferWindowMemory').
        """
        uploaded_files = st.sidebar.file_uploader(label='Upload PDF files', type=['pdf'], accept_multiple_files=True)
        if not uploaded_files:
            st.error("Please upload PDF documents to continue!")
            st.stop()

        user_query = st.chat_input(placeholder="Ask me anything!")

        if uploaded_files and user_query:
            qa_chain = self.create_cr_chain(uploaded_files,memory,window_num)
            st.session_state.messages.append({"role": "user", "content": user_query})
            show_message(user_query, 'user')
            with st.chat_message("assistant"):
                st_cb = StreamlitCallbackHandler(st.container())
                result = qa_chain.invoke(
                    {"question":user_query,},
                    {"callbacks": [st_cb]},
                    
                )
                response = result["answer"]
                st.session_state.messages.append({"role": "assistant", "content": response})

                for idx, doc in enumerate(result['source_documents'],1):
                    filename = os.path.basename(doc.metadata['source'])
                    page_num = doc.metadata['page']
                    ref_title = f":blue[Reference {idx}: *{filename} - page.{page_num}*]"
                    with st.popover(ref_title):
                        st.caption(doc.page_content)