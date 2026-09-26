import fitz
from pathlib import Path
from langchain_core.documents import Document
PDF_FOLDER = Path(__file__).resolve().parents[2] / "pdfs"
def load_pdfs():
    documents = []
    for pdf_path in PDF_FOLDER.glob("*.pdf"):
        print(f"Processing: {pdf_path.name}")
        pdf = fitz.open(pdf_path)
        for page_number, page in enumerate(pdf):
            text = page.get_text("text").strip()
            if not text:
                continue
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": pdf_path.name,
                        "page": page_number + 1,
                    }
                )
            )
        pdf.close()
    print(f"\nTotal pages extracted: {len(documents)}")
    return documents