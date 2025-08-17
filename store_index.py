from src.hepler import load_pdf_file, text_split, download_embeddings
from pinecone import ServerlessSpec 
from pinecone.grpc import PineconeGRPC as Pinecone
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
import os


load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

extracted_data=load_pdf_file(data='Data/')
texts_chunk = text_split(extracted_data)
embedding = download_embeddings()


from pinecone import Pinecone 
pinecone_api_key = PINECONE_API_KEY

pc = Pinecone(api_key=pinecone_api_key)

index_name = "medical-chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name = index_name,
        dimension=384,  # Dimension of the embeddings
        metric= "cosine",  # Cosine similarity
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )


index = pc.Index(index_name)



docsearch = PineconeVectorStore.from_documents(
    documents=texts_chunk,
    embedding=embedding,
    index_name=index_name
)