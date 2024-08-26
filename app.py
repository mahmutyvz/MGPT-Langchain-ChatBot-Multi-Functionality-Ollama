import streamlit as st
import warnings
import datetime

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Langchain",
                   page_icon="chart_with_upwards_trend", layout="wide")
st.markdown("<h1 style='text-align:center;'>Langchain</h1>", unsafe_allow_html=True)
st.write(datetime.datetime.now(tz=None))
tabs = ["Chatbot", "About"]
page = st.sidebar.radio("Tabs", tabs)