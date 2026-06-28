import os
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Set page configuration
st.set_page_config(page_title="TN Ag-Scheme Advisor", page_icon="🌾", layout="centered")

st.title("🌾 TN Agricultural Schemes Advisor")
st.write("Ask questions about the Tamil Nadu Welfare Agricultural Schemes dynamically.")

# Check for API Key
if not os.environ.get("OPENAI_API_KEY"):
    st.error("Please set your OPENAI_API_KEY environment variable in your terminal!")
    st.stop()

# Cache the vector database setup so it doesn't reload and re-download on every click
@st.cache_resource
def initialize_rag():
    url = "https://www.tn.gov.in/scheme_list.php?dep_id=Mg=="
    loader = WebBaseLoader(url)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # Using an ephemeral/in-memory Chroma store for quick dashboard reloads
    vector_db = Chroma.from_documents(documents=chunks, embedding=embeddings)
    return vector_db

# Initialize pipeline
with st.spinner("Initializing RAG Pipeline (Loading webpage & embeddings)..."):
    vector_db = initialize_rag()

# Initialize LLM and Prompt
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant specialized in Tamil Nadu Agricultural Schemes.\n"
               "Answer the user's question using ONLY the provided context below. If you don't know the answer, say you don't know.\n\n"
               "Context:\n{context}"),
    ("human", "{question}")
])

# Streamlit Input UI
user_query = st.text_input("Enter your question here:", placeholder="e.g., Is there any scheme for pest management?")

if user_query:
    with st.spinner("Searching database and thinking..."):
        # 1. Retrieve
        matching_docs = vector_db.similarity_search(user_query, k=2)
        context_text = "\n\n".join([doc.page_content for doc in matching_docs])
        
        # 2. Format Prompt
        formatted_prompt = prompt_template.format_messages(
            context=context_text,
            question=user_query
        )
        
        # 3. Generate
        response = llm.invoke(formatted_prompt)
        
    # Display results nicely
    st.subheader("💡 Answer:")
    st.write(response.content)
    
    # Optional: Expandable section to see what source information was used
    with st.expander("📚 View Retreived Context Chunks"):
        for i, doc in enumerate(matching_docs):
            st.markdown(f"**Chunk #{i+1}:**")
            st.code(doc.page_content)