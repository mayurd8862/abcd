import streamlit as st
from streamlit_chat import message
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import tempfile
import asyncio
import nest_asyncio

# Fix: allow nested event loops inside Streamlit
nest_asyncio.apply()

st.title("📑🤖 Chat with Data")

# Groq LLM setup
groq_api_key = st.secrets["GROQ_API_KEY"]
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")

# Cache vectorstore so it doesn’t reload every time
@st.cache_resource
def load_and_process_data(files):
    all_splits = []
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file.read())
            file_path = temp_file.name

        loader = PyPDFLoader(file_path)
        data = loader.load_and_split()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=100
        )
        splits = text_splitter.split_documents(data)
        all_splits.extend(splits)

    # Embeddings (Google)
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_db = FAISS.from_documents(all_splits, embedding)
    return vector_db

# Async response generator
async def response_generator(vector_db, query):
    template = """Use the following context to answer the question.
If you don’t know the answer, just say you don’t know.
Keep the answer concise (max 3 sentences).

Context: {context}
Question: {question}
Helpful Answer:"""

    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vector_db.as_retriever(),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )

    # Run chain in thread (avoids blocking event loop)
    result = await asyncio.to_thread(qa_chain, {"query": query})
    return result["result"]

# --- UI Section ---
files = st.file_uploader("Upload PDF File(s)", type=["pdf"], accept_multiple_files=True)
submit_pdf = st.checkbox("Submit and chat (PDF)")

if files and submit_pdf:
    vector_db = load_and_process_data(files)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User input
    if query := st.chat_input("Ask a question about your PDF..."):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.spinner("Generating response..."):
            # Await response without asyncio.run()
            response = asyncio.run(response_generator(vector_db, query))

        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
