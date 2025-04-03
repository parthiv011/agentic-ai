from fastapi import FastAPI
from pydantic import BaseModel
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = OllamaLLM(model="llama3.2")

template = """
You are an expert in technical knowledge of software solutions.

Here are some success stories/insights of how you have helped companies with software solutions: {blogData}

Here are the questions to answer: {question}
"""
prompt = ChatPromptTemplate.from_template(template=template)
chain = prompt | model

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(request: QueryRequest):
    """Endpoint to answer a user's question using the vector database and LLM"""
    blogData = retriever.invoke(request.question) or "No relevant case studies found."

    response = chain.invoke({"blogData": blogData, "question": request.question})

    answer = response if isinstance(response, str) else getattr(response, "content", str(response))

    return {"answer": answer}

# Run using: uvicorn app:app --reload
