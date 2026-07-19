from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()


documents = [
    Document(
        page_content="Dogs are loyal, friendly animals and make great companions.",
        metadata={"source": "mammal-pets"},
    ),

    Document(
        page_content="Cats are independent animals and usually enjoy spending time in their own space.",
        metadata={"source": "mammal-pets"},
    ),

    Document(
        page_content="Goldfish are popular pets for beginners because they are relatively easy to care for.",
        metadata={"source": "fish-pets"},
    ),

    Document(
        page_content="Parrots are intelligent birds that can imitate human speech.",
        metadata={"source": "bird-pets"},
    ),

    Document(
        page_content="Rabbits are social animals and need plenty of space to move around.",
        metadata={"source": "mammal-pets"},
    ),
]


embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model
)


results = vector_store.similarity_search(
    "Which pet is loyal and friendly?",
    k=1
)


print(results[0].page_content)
print(results[0].metadata)