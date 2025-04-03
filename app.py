from fastapi import FastAPI
from pydantic import BaseModel
from langchain_ollama.llms import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from vector import retriever
from retrieval_grader import is_relevant  # Import grading function
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize QA Model
qa_model = OllamaLLM(model="llama3.2")

# QA Prompt Template
qa_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|> 
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know.

Use an appropriate length of your choice to answer the question but keep the answer concise.

<|eot_id|><|start_header_id|>user<|end_header_id|>
Question: {question}
Context: {context}
Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

qa_prompt = ChatPromptTemplate.from_template(template=qa_template)
qa_chain = qa_prompt | qa_model

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(request: QueryRequest):
    """Endpoint to answer a user's question using the vector database and LLM"""

    # Retrieve documents
    docs = retriever.invoke(request.question)

    if not docs:
        return {"answer": "No relevant information found."}

    # Combine retrieved docs into a single text block
    doc_txt = "\n\n".join([doc.page_content for doc in docs])

    # Check relevance before generating an answer
    if not is_relevant(request.question, doc_txt):
        return {"answer": "No relevant information found to answer this question."}

    # Generate an answer
    response = qa_chain.invoke({"question": request.question, "context": doc_txt})

    # Ensure response is properly formatted
    answer = response if isinstance(response, str) else getattr(response, "content", str(response))

    return {"answer": answer}

# Run using: uvicorn app:app --reload
