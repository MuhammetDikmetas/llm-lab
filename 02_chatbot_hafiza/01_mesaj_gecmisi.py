from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

model = ChatAnthropic(model="claude-sonnet-4-5", temperature=0.1)

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

prompt = ChatPromptTemplate.from_messages([
    ("system", "Sen yardımcı bir asistansın. Kısa ve anlaşılır cevaplar ver."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

chain = prompt | model

chatbot = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history"
)

config = {"configurable": {"session_id": "abcde123"}}

while True:
    user_input = input("Sen: ").strip()

    if user_input.lower() in {"çıkış", "cikis", "exit"}:
        print("Sohbet sonlandırıldı.")
        break

    response = chatbot.invoke({"question": user_input}, config=config)
    print("Claude:", response.content)