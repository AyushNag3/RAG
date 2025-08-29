## RAG Pipeline with Functions

from dotenv import load_dotenv
load_dotenv()
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever


# ----------------- Functions -----------------

def load_pdf(doc_path: str):
    """Load PDF and return documents."""
    if doc_path and os.path.exists(doc_path):
        loader = PyPDFLoader(doc_path)
        data = loader.load()
        print("✅ Done loading PDF....")
        return data
    else:
        raise FileNotFoundError("❌ PDF file not found.")


def split_into_chunks(data, chunk_size=1200, chunk_overlap=300):
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(data)
    print("✅ Done splitting into chunks....")
    return chunks


def create_vector_db(chunks):
    """Create Chroma vector database from chunks using HuggingFace embeddings."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="simple-rag",
    )
    print("✅ Done creating vector database....")
    return vector_db


def create_llm():
    """Initialize Groq LLM."""
    return ChatGroq(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        api_key=os.getenv("GROQ_API_KEY")
    )


def create_retriever(vector_db, llm):
    """Create multi-query retriever to generate variations of user query."""
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
    return retriever


def create_chain(retriever, llm):
    """Build the full RAG chain."""
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
    return chain


def run_query(chain, query: str):
    """Run a query against the chain."""
    res = chain.invoke(query)
    return res


# ----------------- Main -----------------

def main():
    doc_path = "./data/BOI.pdf"

    # Step 1. Load PDF
    data = load_pdf(doc_path)

    # Step 2. Split into chunks
    chunks = split_into_chunks(data)

    # Step 3. Create vector DB
    vector_db = create_vector_db(chunks)

    # Step 4. Initialize LLM
    llm = create_llm()

    # Step 5. Create retriever
    retriever = create_retriever(vector_db, llm)

    # Step 6. Create chain
    chain = create_chain(retriever, llm)

    # Step 7. Run query
    query = input("Enter your query: ")
    res = run_query(chain, query)

    print("\n✅ Final Answer:\n", res)


if __name__ == "__main__":
    main()
