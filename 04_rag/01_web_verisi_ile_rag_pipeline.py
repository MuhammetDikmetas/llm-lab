import bs4
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()


llm = ChatAnthropic(
    model="claude-sonnet-4-5",
    temperature=0.1
)


loader = WebBaseLoader(
    web_paths=(
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
    ),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)

docs = loader.load()


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

splits = text_splitter.split_documents(docs)


embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embedding_model
)


retriever = vectorstore.as_retriever()



prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, just say that you don't know. "
            "Use three sentences maximum and keep the answer concise.\n"
            "Question: {question} \nContext: {context} \nAnswer:",
        )
    ]
)


def format_docs(docs):
    return "\n\n".join(
        doc.page_content for doc in docs
    )


rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)


if __name__ == "__main__":
    for chunk in rag_chain.stream(
        "What is Task Decomposition?"
    ):
        print(
            chunk,
            end="",
            flush=True
        )