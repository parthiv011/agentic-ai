from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd

# Load data
df = pd.read_json('knowledge/blogs.json')

# Initialize embeddings model
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Define ChromaDB location
db_location = './chroma_langchain_db'

# Initialize ChromaDB
vector_store = Chroma(
    collection_name="success_stories",
    persist_directory=db_location,
    embedding_function=embeddings
)

# Check if documents need to be added
add_documents = vector_store._collection.count() == 0

if add_documents:
    documents = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)

    # Process each row in the dataset
    for _, row in df.iterrows():
        meta = row["meta"]
        metadata = {
            "author": meta.get("author", "Unknown"),
            "date": meta.get("date", "Unknown"),
            "category": meta.get("category", "Uncategorized"),
            "keywords": meta.get("keywords", ""),
            "slug": meta.get("slug", ""),
            "references": ", ".join(meta.get("refer", [])),
            "title": meta.get('title', 'Untitled'),
            "subtitle": meta.get('subtitle', ''),
            "description": meta.get('description', ''),
        }

        # Split content into chunks
        full_text = row.get('content', '')
        chunks = splitter.split_text(full_text)

        for i, chunk in enumerate(chunks):
            doc_id = f"{row.name}_{i}"
            documents.append(Document(page_content=chunk, metadata=metadata, id=doc_id))

    print(f"✅ Total documents to add: {len(documents)}")

    batch_size = 500
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        print(f"🔄 Adding batch {i // batch_size + 1} with {len(batch)} documents...")
        vector_store.add_documents(documents=batch, ids=[doc.id for doc in batch])

    vector_store._client.persist()
    print("✅ Documents successfully stored!")

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# stored_count = vector_store._collection.count()
# print(f"🔍 Vector store now contains {stored_count} documents!")

# sample_data = vector_store._collection.get(limit=5)  
# print(f"🔍 Sample stored documents:\n{sample_data}")

print("🚀 Vector store initialized & ready for queries!")