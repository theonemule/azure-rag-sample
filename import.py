from langchain_text_splitters import TokenTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector

connection = "" #Connection String
  
collection_name = "citus"

embeddings = AzureOpenAIEmbeddings(model="embeddingmodel")

text_splitter = TokenTextSplitter(
    # Controls the size of each chunk
    chunk_size=2000,
    # Controls overlap between chunks
    chunk_overlap=20,
)

vectorstore = PGVector(
    embeddings=embeddings,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True,
)

vectorstore.drop_tables()
vectorstore.create_tables_if_not_exists()
vectorstore.create_collection()

# Load and split the text document.
with open('sotd.txt', 'r') as file:
    file_contents = file.read()
   
texts = text_splitter.split_text(file_contents)

# Create embeddings for each of the documents
docs = text_splitter.create_documents(texts)

for idx, doc in enumerate(docs):
    doc.metadata["id"] = idx

vectorstore.add_documents(docs, ids=[doc.metadata["id"] for doc in docs])