from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="llama3.2")

template = """
You are an expert in technical knowledge of softwares

Here are some success stories/insights of how you have helped the companies with software solutions: {blogData}

Here are the questions to answer: {question}
"""

prompt = ChatPromptTemplate.from_template(template=template)
chain = prompt | model

while True:
    print("\n\n-----------------------------------------------------------")
    question = input("Ask your question: (q to quit): ")
    print("\n\n")

    if question == 'q':
        break

    blogData = retriever.invoke(question)
    result = chain.invoke({"blogData": blogData, "question": question})

    print(result)