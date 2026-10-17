import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from search import get_embeddings, get_store

load_dotenv()

PDF_PATH = os.getenv("PDF_PATH")

BATCH_SIZE = 20
MAX_RETRIES = 5
RETRY_WAIT_SECONDS = 65


def add_documents_with_retry(store, batch):
    for attempt in range(MAX_RETRIES):
        try:
            store.add_documents(batch)
            return
        except Exception as e:
            is_rate_limit = "429" in str(e) or "ResourceExhausted" in str(e)
            if is_rate_limit and attempt < MAX_RETRIES - 1:
                print(f"Rate limit atingido, aguardando {RETRY_WAIT_SECONDS}s antes de tentar novamente...")
                time.sleep(RETRY_WAIT_SECONDS)
            else:
                raise


def ingest_pdf():
    embeddings = get_embeddings()

    if not embeddings:
        print("Nenhuma API key configurada no .env. Defina OPENAI_API_KEY ou GOOGLE_API_KEY.")
        return

    loader = PyPDFLoader(PDF_PATH)
    documentos = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documentos)

    for chunk in chunks:
        chunk.metadata = {k: v for k, v in chunk.metadata.items() if v not in ("", None)}

    store = get_store(embeddings)

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        add_documents_with_retry(store, batch)
        print(f"Gravados {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)} chunks.")

    print(f"Ingestão concluída: {len(chunks)} chunks gravados no banco.")


if __name__ == "__main__":
    ingest_pdf()
