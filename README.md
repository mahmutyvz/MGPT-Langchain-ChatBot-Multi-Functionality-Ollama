# Langchain Streamlit with Ollama
### Project Scructure

```
MGPT-Langchain-ChatBot-Multi-Functionality-Ollama/
├─ src/
├─ .gitignore
├─ requirements.txt
├─ README.md
```

The root variable in paths.py must be changed to the absolute path of the project.

### Install the required dependencies.
```shell
Python 3.10.11 or higher is required.
Ollama must be installed on your machine for optimal functionality. Link : https://ollama.com/download
pip install -r requirements.txt
```
### MGPT Features
Here are various ways to utilize the chatbot developed with LangChain and Streamlit:

-  **Memory-Enabled Chatbot** \
   A chatbot that retains the context of previous interactions to deliver more relevant responses.

-  **Web-Connected Chatbot** \
   A chatbot equipped to browse the web and provide answers to queries about the latest news and events.

-  **Document-Based Chatbot** \
   Enable the chatbot to read and interpret user-provided documents, allowing it to answer questions based on specific content.

-  **SQL Database Chatbot** \
   Interact with SQL databases using conversational language, enabling users to easily create and execute queries.

-  **Website Interaction Chatbot** \
   Empower the chatbot to access and retrieve information from a given website URL to answer questions based on its content.

### Running the Application

You can directly run the application, make training and predictions. 

```bash
streamlit run app.py
```  
