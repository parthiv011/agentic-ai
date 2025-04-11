llama lightweight model
mxbai-embed-large - for embeddings of resources in the vector db.

Challenges
1. Incomplete/messy data
2. Inacurrate Retrieval from the vector db
3. Mainly depends on how the question is framed/asked (doesnt really work for complex questions)
4. doesnt study from the history of the chat

How to improve this
1. Better parsing of data
2. proper chunk size
3. Rerank the vectors for relevant retrieval
4. hybrid the search process
    vector search + keyword search


from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType

def multiply(a: int, b: int) -> int:
    print("TOOL EXECUTED")
    return a * b

def multiply_tool_wrapper(input_data):
    try:
        # Handle string like "(12, 7)"
        if isinstance(input_data, str):
            input_data = input_data.strip().replace("(", "").replace(")", "").replace(",", "")
            parts = input_data.split()
            a, b = int(parts[0]), int(parts[1])
        elif isinstance(input_data, (tuple, list)):
            a, b = int(input_data[0]), int(input_data[1])
        elif isinstance(input_data, dict):
            a, b = int(input_data.get("a")), int(input_data.get("b"))
        else:
            return "Error: Unsupported input format"

        result = a * b
        print("TOOL EXECUTED")
        return f"The result of {a} x {b} is {result}"
    except Exception as e:
        return f"Error: {e}"

multiply_tool = Tool.from_function(
    func=multiply_tool_wrapper,
    name="multiply",
    description="Multiply two integers. Input should be like '(4, 5)' or '4 5'."
)

agent = initialize_agent(
    tools=[multiply_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

prompt = "Can u tell me a joke?"
response = agent.invoke({"input": prompt})
print("Agent Response:", response["output"])