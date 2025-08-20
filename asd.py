import streamlit as st
from streamlit_chat import message
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings # Added this missing import
import tempfile
import asyncio
import os

# --- Main App Configuration ---
st.title("📄 Chat with Data")

# --- Model and API Initialization ---
groq_api_key = st.secrets["GROQ_API_KEY"]
# Note: You'll also need a GOOGLE_API_KEY in your secrets for the embeddings
# os.environ['GOOGLE_API_KEY'] = st.secrets["GOOGLE_API_KEY"]

llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")

# --- Data Loading and Processing ---
@st.cache_resource
def load_and_process_data(files):
    all_splits = []
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file.read())
            file_path = temp_file.name
        
        loader = PyPDFLoader(file_path)
        data = loader.load_and_split()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
        splits = text_splitter.split_documents(data)
        all_splits.extend(splits)
    
    # Embedding model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Create vector database
    vectordb = FAISS.from_documents(all_splits, embeddings)
    return vectordb

# --- Asynchronous Response Generation ---
async def response_generator(vectordb, query):
    template = """
    Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    Use three sentences maximum. Keep the answer as concise as possible.
    
    {context}
    
    Question: {question}
    
    Helpful Answer:"""
    
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    
    # Use the modern .ainvoke() for async calls
    result = await qa_chain.ainvoke({"query": query})
    return result["result"]

# --- Streamlit UI ---
files = st.file_uploader("Upload PDF file(s)", type=["pdf"], accept_multiple_files=True)
submit = st.button("Submit and Chat (PDF)")

st.markdown("---")

if submit and files:
    with st.spinner("Processing documents..."):
        st.session_state.vectordb = load_and_process_data(files)
        st.success("Documents processed! You can now ask questions.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
        
    if "vectordb" in st.session_state:
        vectordb = st.session_state.vectordb
        with st.spinner("Generating response..."):
            
            # *FIXED ASYNCIO HANDLING*
            # This robustly gets or creates an event loop for the current thread
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Run the async function using the event loop
            response = loop.run_until_complete(response_generator(vectordb, query))

            with st.chat_message("assistant"):
                st.markdown(response)
            
            st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.warning("Please upload and submit documents first.")
