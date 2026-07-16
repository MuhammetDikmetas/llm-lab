from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

model = ChatAnthropic(
    model="claude-sonnet-4-5",
    temperature=0.1
)

messages = [
    SystemMessage(
        content="Sen deneyimli bir AI mühendisisin. Kısa, doğru ve anlaşılır cevaplar ver."
    ),
    HumanMessage(
        content="AI Engineer olmak isteyen bir bilgisayar mühendisliği öğrencisi hangi adımları izlemelidir?"
    )
]

response = model.invoke(messages)

print(response.content)