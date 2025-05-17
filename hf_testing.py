from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import streamlit as st         #use to view output in much better way ,provide us interface for using AIML
import os 
from langchain_groq import ChatGroq 
load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"] = "true"     #to store records in langsmith

HF_TOKEN = os.getenv("HF_TOKEN")

st.title("Langchain joke generator")
st.markdown("Powered by Hugging Face")

topic = st.text_input("Enter topic for the joke ")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a joke generating assistant. Generate only ONE joke for the given topic don't continue the conversation"),
        ("user","topic:{topic}")
    ]
)
llm = HuggingFaceEndpoint(
    endpoint_url ="https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta",
    huggingfacehub_api_token = HF_TOKEN
)
model = ChatGroq(model="Gemma2-9b-It")
output_parser = StrOutputParser()
chain = prompt|llm|output_parser
if topic:
    with st.spinner("Generating your joke..."):
        response = chain.invoke({"topic" : topic})
        st.success("Here is your generated joke")
        st.write(response.strip())