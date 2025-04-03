from langchain_ollama.llms import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Initialize Grader Model
grader_model = OllamaLLM(model="llama3.2", format="json", temperature=0)

# Grader Prompt Template
grader_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|> 
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

grader_prompt = PromptTemplate(template=grader_template, input_variables=["question", "document"])
retrieval_grader = grader_prompt | grader_model | JsonOutputParser()

def is_relevant(question: str, document: str) -> bool:
    """Returns True if the document is relevant to the question, False otherwise."""
    grader_output = retrieval_grader.invoke({"question": question, "document": document})
    return grader_output.get("score") == "yes"
