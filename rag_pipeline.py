import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
# Import OpenAI LLM and ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Ensure API Key is loaded
if not os.environ.get("OPENAI_API_KEY"):
    print("[Error] Please set your OPENAI_API_KEY environment variable!")
    exit()

# --- Step 1: Load Document ---
url = "https://www.tn.gov.in/scheme_list.php?dep_id=Mg=="
loader = WebBaseLoader(url)
documents = loader.load()

# --- Step 2: Split Document ---
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

# --- Step 3: Embeddings & Vector Store ---
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma.from_documents(documents=chunks, embedding=embeddings)

# --- Step 4: LLM Generation ---
# Initialize OpenAI Chat Model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Create a prompt template instructing the AI how to behave
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant specialized in Tamil Nadu Agricultural Schemes.\n"
               "Answer the user's question using ONLY the provided context below. If you don't know the answer, say you don't know.\n\n"
               "Context:\n{context}"),
    ("human", "{question}")
])

# Define our User Query
user_query = "What kind of training is available for farmers according to the scheme list?"

print(f"\nUser Query: '{user_query}'")
print("Retrieving context from database...")

# 1. Retrieve matching documents
matching_docs = vector_db.similarity_search(user_query, k=2)
# Combine the text from the matching chunks
context_text = "\n\n".join([doc.page_content for doc in matching_docs])

# 2. Format the prompt with our retrieved data
formatted_prompt = prompt_template.format_messages(
    context=context_text,
    question=user_query
)

print("Generating answer using OpenAI...")
# 3. Ask the LLM
response = llm.invoke(formatted_prompt)

print("\n=== AI RESPONSE ===")
print(response.content)
print("====================")