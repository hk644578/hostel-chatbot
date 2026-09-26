from hostel_project.data_extract import load_pdfs
from langchain_text_splitters import RecursiveCharacterTextSplitter
from hostel_project.embeddings import get_embeddings
documents = load_pdfs()
print(f"Documents: {len(documents)}")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=75
)
chunks = splitter.split_documents(documents)
print(f"Chunks: {len(chunks)}")
texts = [chunk.page_content for chunk in chunks]
embeddings = get_embeddings(texts)
print(f"Embeddings: {len(embeddings)}")
print(f"Embedding dimension: {len(embeddings[0])}")