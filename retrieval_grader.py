from langchain_ollama.llms import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from vector import retriever

model = OllamaLLM(model="llama3.2", format="json", temperature=0)

template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|> 
You are a grader assessing relevance of a retrieved document to a user question. 
If the document contains keywords related to the user question, grade it as relevant. 
It does not need to be a stringent test. The goal is to filter out erroneous retrievals. 

Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. 
Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.

<|eot_id|><|start_header_id|>user<|end_header_id|>
Here are the retrieved documents: \n\n {document} \n\n
Here is the user question: {question} \n 
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

prompt = PromptTemplate(template=template, input_variables=["question", "document"])
retrieval_grader = prompt | model | JsonOutputParser()

question = "Can you brief about the challenges had under D2C apparels?"

docs = retriever.invoke(question)

print("Retrieved Documents:")
for i, doc in enumerate(docs):
    print(f"Doc {i+1}: {doc.page_content}")  # Print first 300 chars

# Check if there are documents
if not docs:
    print("No documents retrieved!")
else:
    # Join all retrieved docs into a single text block
    doc_txt = "\n\n".join([doc.page_content for doc in docs])

    # Invoke the model with the properly formatted document text
    result = retrieval_grader.invoke({"question": question, "document": doc_txt})

    print("Grader Output:", result)
