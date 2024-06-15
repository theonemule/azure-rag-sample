from flask import Flask, request, jsonify, send_from_directory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_postgres.vectorstores import PGVector
from langchain_openai import AzureOpenAI, AzureOpenAIEmbeddings


app = Flask(__name__)

# Initialize the OpenAI language model for response generation
llm = AzureOpenAI(
    deployment_name="completionmodel",
)

# Initialize the embedding function
embeddings = AzureOpenAIEmbeddings(model="embeddingmodel")

# Initialize the vector database

connection = ""  

collection_name = "citus"

vectorstore = PGVector(
    embeddings=embeddings,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True,
)

# Define the prompt template for generating AI responses
PROMPT_TEMPLATE = """
Human: You are a research assistant, and provides answers to questions about the story.

Use the following pieces of information to provide a concise answer to the question enclosed in <question> tags.

If you don't know the answer, just say that you don't know, don't try to make up an answer.

<context>
{context}
</context>

<question>
{question}
</question>

The response should be specific.
"""

# Create a PromptTemplate instance with the defined template and input variables
prompt = PromptTemplate(
    template=PROMPT_TEMPLATE, input_variables=["context", "question"]
)
# Convert the vector store to a retriever
retriever = vectorstore.as_retriever()

# Define a function to format the retrieved documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Define the RAG (Retrieval-Augmented Generation) chain for AI response generation
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')
    

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question', '')

    # queue up a query to ask the RAG app
    docs = vectorstore.similarity_search(question, k=3)

    # Invoke the RAG chain with a specific question and retrieve the response
    res = rag_chain.invoke(question)
    
    payload = {'response': res}

    
    return jsonify(payload)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)