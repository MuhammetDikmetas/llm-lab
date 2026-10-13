import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


PERSIST_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".chroma")
COLLECTION_NAME = "rag_chroma"

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def build_vector_store() -> Chroma:
    """Web sayfalarını indirir, parçalara böler ve Chroma'ya kaydeder."""
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs_list)

    return Chroma.from_documents(
        documents=splits,
        collection_name=COLLECTION_NAME,
        embedding=embedding_model,
        persist_directory=PERSIST_DIRECTORY,
    )



if not os.path.exists(PERSIST_DIRECTORY):
    print("---VEKTÖR VERİTABANI OLUŞTURULUYOR---")
    build_vector_store()

retriever = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embedding_model,
).as_retriever()


if __name__ == "__main__":
    results = retriever.invoke("What is agent memory?")
    print(f"{len(results)} belge bulundu")
    print(results[0].page_content[:500])
