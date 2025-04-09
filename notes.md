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