## 1. Ingest PDF Files
# 2. Extract Text from PDF Files and split into small chunks
# 3. Send the chunks to the embedding model
# 4. Save the embeddings to a vector database
# 5. Perform similarity search on the vector database to find similar documents
# 6. Retrieve the similar documents and present them to the user
## run pip install -r requirements.txt to install the required packages
from dotenv import load_dotenv
load_dotenv()
import os
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import OnlinePDFLoader # loads pdf from url 
from langchain_community.document_loaders import PyPDFLoader

doc_path = "./data/BOI.pdf"
data = None
# Local PDF file uploads
if doc_path and os.path.exists(doc_path):
    loader = PyPDFLoader(doc_path)
    data = loader.load()
    print("✅ Done loading PDF....")
else:
    print("Upload a PDF file")

# Preview first page
content = data[0].page_content
print(content[:200])
# print(content[:100])

# ==== End of PDF Ingestion ====


# ==== Extract Text from PDF Files and Split into Small Chunks ====

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Split and chunk
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
chunks = text_splitter.split_documents(data)
print("✅ Done splitting....")

# ===== Add to vector database ===
# Using HuggingFace embeddings since Groq doesn’t provide embedding models
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="simple-rag",
   
)

print("✅ Done adding to vector database....")


## === Retrieval ===
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

# set up our model to use Groq
llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    api_key=os.getenv("GROQ_API_KEY")  # or use environment variable
)

# multi-query retriever
QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI assistant. Generate five different versions of the given user question
    to retrieve relevant documents from a vector database. 
    Provide these alternative questions separated by newlines.
    Original question: {question}""",
)

retriever = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
)

# RAG prompt
template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Run query
query = input("Enter your query : ")
res = chain.invoke(query)

print("✅ Final Answer:\n", res)
