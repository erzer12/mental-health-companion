import os
import json
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Global paths
DATA_PATH = "data"
DB_FAISS_PATH = "vectorstore/db_faiss"
MANIFEST_PATH = "vectorstore/manifest.json" # Keeps track of what we learned

def get_current_files():
    """Returns a sorted list of PDF filenames in the data folder."""
    if not os.path.exists(DATA_PATH):
        return []
    return sorted([f for f in os.listdir(DATA_PATH) if f.endswith('.pdf')])

def build_knowledge_base():
    """
    Smart Build: Only rebuilds if the PDFs have changed.
    """
    current_files = get_current_files()
    
    # --- 1. THE FRESHNESS CHECK ---
    if os.path.exists(DB_FAISS_PATH) and os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, 'r') as f:
            last_indexed_files = json.load(f)
        
        if current_files == last_indexed_files:
            print("✅ Database is up to date. Loading...")
            return load_knowledge_base()
        else:
            print("🔄 Changes detected in PDF library. Rebuilding database...")
    else:
        print("⚠️ No database found. Building from scratch...")

    # --- 2. BUILD PROCESS ---
    if not current_files:
        print("❌ No PDF files found in 'data/'")
        return None

    print(f"📚 Ingesting {len(current_files)} files: {current_files}")
    
    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()
    
    # Split Text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    # Embed & Save
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(texts, embeddings)
    
    # SAFETY: Ensure folder exists before saving
    if not os.path.exists("vectorstore"):
        os.makedirs("vectorstore")

    db.save_local(DB_FAISS_PATH)
    
    # SAVE MANIFEST
    with open(MANIFEST_PATH, 'w') as f:
        json.dump(current_files, f)
    
    print(f"✅ Rebuild Complete. Indexed {len(texts)} chunks.")
    return db

def load_knowledge_base():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    try:
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        print(f"Error loading DB: {e}")
        return None

def retrieve_context(db, query, k=2):
    if db is None: return ""
    results = db.similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in results])