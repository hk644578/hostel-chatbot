import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_chroma import Chroma


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

PDF_FOLDER = BASE_DIR / "web_data"
CHROMA_DIR = BASE_DIR / "chroma_db"

ENV_FILE = BASE_DIR / ".env"


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

print("Loading environment variables...")

load_dotenv(ENV_FILE, override=True)

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

if not NVIDIA_API_KEY:
    raise ValueError(
        f"NVIDIA_API_KEY was not found.\n"
        f"Please check your .env file:\n{ENV_FILE}"
    )

print("NVIDIA API key found.")
print(f"API key prefix: {NVIDIA_API_KEY[:6]}")
print(f".env location: {ENV_FILE}")


# ============================================================
# LOAD PDF FILES
# ============================================================

def load_pdfs():

    documents = []

    if not PDF_FOLDER.exists():
        raise FileNotFoundError(
            f"PDF folder does not exist:\n{PDF_FOLDER}"
        )

    pdf_files = sorted(PDF_FOLDER.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found inside:\n{PDF_FOLDER}"
        )

    print()
    print(f"Found {len(pdf_files)} PDF files.")

    for pdf_file in pdf_files:

        print(f"Loading: {pdf_file.name}")

        try:

            loader = PyPDFLoader(str(pdf_file))

            docs = loader.load()

            for doc in docs:
                doc.metadata["source"] = pdf_file.name

            documents.extend(docs)

        except Exception as e:

            print(f"ERROR loading {pdf_file.name}")
            print(f"Reason: {e}")

    if not documents:
        raise ValueError("No PDF pages were successfully loaded.")

    print()
    print(f"Loaded {len(documents)} pages.")

    return documents


# ============================================================
# SPLIT DOCUMENTS
# ============================================================

def split_documents(documents):

    print()
    print("Splitting documents into chunks...")

    text_splitter = RecursiveCharacterTextSplitter(

        chunk_size=1000,

        chunk_overlap=200,

        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = text_splitter.split_documents(documents)

    if not chunks:
        raise ValueError("No chunks were created from the PDFs.")

    print(f"Created {len(chunks)} chunks.")

    return chunks


# ============================================================
# CREATE NVIDIA EMBEDDINGS
# ============================================================

def create_embeddings():

    print()
    print("========== NVIDIA EMBEDDINGS ==========")

    print("Loading NVIDIA embedding model...")

    embeddings = NVIDIAEmbeddings(

        model="nvidia/nemotron-3-embed-1b",

        api_key=NVIDIA_API_KEY
    )

    print("NVIDIA embedding model loaded.")

    # --------------------------------------------------------
    # TEST NVIDIA BEFORE CHROMADB
    # --------------------------------------------------------

    print()
    print("Testing NVIDIA embedding API...")

    test_text = "What is the hostel fee at NIT Jalandhar?"

    try:

        test_embedding = embeddings.embed_query(test_text)

        print("NVIDIA embedding test successful.")

        print(
            f"Embedding dimensions: {len(test_embedding)}"
        )

    except Exception as e:

        print()
        print("NVIDIA EMBEDDING TEST FAILED")
        print("--------------------------------")
        print(e)
        print("--------------------------------")

        raise

    return embeddings


# ============================================================
# CREATE CHROMADB
# ============================================================

def create_vector_database(chunks, embeddings):

    print()
    print("========== CHROMADB ==========")

    print(f"ChromaDB location: {CHROMA_DIR}")

    # --------------------------------------------------------
    # REMOVE OLD DATABASE
    # --------------------------------------------------------
    #
    # IMPORTANT:
    # Uncomment the following lines if you want to create
    # the database from scratch every time.
    #
    # --------------------------------------------------------

    if CHROMA_DIR.exists():

        print()
        print("Existing ChromaDB found.")

        print("Removing old ChromaDB...")

        try:

            shutil.rmtree(CHROMA_DIR)

            print("Old ChromaDB removed.")

        except Exception as e:

            raise RuntimeError(
                f"Could not remove old ChromaDB:\n{e}"
            )

    # --------------------------------------------------------
    # CREATE NEW CHROMA DATABASE
    # --------------------------------------------------------

    print()
    print("Creating new ChromaDB...")

    try:

        vectorstore = Chroma.from_documents(

            documents=chunks,

            embedding=embeddings,

            persist_directory=str(CHROMA_DIR),

            collection_name="nitj_documents"
        )

    except Exception as e:

        print()
        print("CHROMADB CREATION FAILED")
        print("--------------------------------")
        print(e)
        print("--------------------------------")

        raise

    print()
    print("Vector database created successfully.")

    print(f"Location: {CHROMA_DIR}")

    return vectorstore


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("================================================")
    print("              RAG INGESTION")
    print("================================================")

    # --------------------------------------------------------
    # STEP 1: LOAD PDFs
    # --------------------------------------------------------

    documents = load_pdfs()

    # --------------------------------------------------------
    # STEP 2: CREATE CHUNKS
    # --------------------------------------------------------

    chunks = split_documents(documents)

    # --------------------------------------------------------
    # STEP 3: CREATE NVIDIA EMBEDDINGS
    # --------------------------------------------------------

    embeddings = create_embeddings()

    # --------------------------------------------------------
    # STEP 4: CREATE CHROMADB
    # --------------------------------------------------------

    vectorstore = create_vector_database(
        chunks,
        embeddings
    )

    # --------------------------------------------------------
    # FINAL INFORMATION
    # --------------------------------------------------------

    unique_sources = set()

    for document in documents:

        source = document.metadata.get(
            "source",
            "unknown"
        )

        unique_sources.add(source)

    print()
    print("================================================")
    print("          INGESTION COMPLETED")
    print("================================================")

    print(f"PDFs processed : {len(unique_sources)}")
    print(f"Pages loaded   : {len(documents)}")
    print(f"Chunks created : {len(chunks)}")
    print(f"Vector DB      : {CHROMA_DIR}")

    print()
    print("Your RAG vector database is ready.")
    print("================================================")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()