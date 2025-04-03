from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

df = pd.read_json('knowledge/blogs.json')

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = './chroma_langchain_db'
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []

    for i, row in df.iterrows():
        meta = row["meta"]
        doc_id = str(i)

        metadata = {
            "author": meta.get("author", "Unknown"),
            "date": meta.get("date", "Unknown"),
            "category": meta.get("category", "Uncategorized"),
            "keywords": meta.get("keywords", ""),  # Can be empty
            "slug": meta.get("slug", ""),
            "references": ", ".join(meta.get("refer", [])), 
        }

        page_content = f"{meta.get('title', 'Untitled')}. " \
               f"{meta.get('subtitle', '')}. " \
               f"{meta.get('description', '')}. " \
               f"{row.get('content', '')[:1000]}" 

        document = Document(page_content=page_content, metadata=metadata, id=doc_id)
        
        documents.append(document)
        ids.append(doc_id)

vector_store = Chroma(
    collection_name="success_stories",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

retriever = vector_store.as_retriever(
    search_type="mmr",  
    search_kwargs={"k": 10, "fetch_k": 25} )

print("Vector store initialized & ready for queries!")
