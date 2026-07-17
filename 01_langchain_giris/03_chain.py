from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatAnthropic(
    model="claude-sonnet-4-5",
    temperature=0.1
)

messages = [
    SystemMessage(
        content="Sen deneyimli bir AI eğitmenisin. Teknik konuları kısa ve anlaşılır açıkla."
    ),
    HumanMessage(
        content="LangChain'de chain yapısı ne işe yarar?"
    )
]

parser = StrOutputParser()

chain = model | parser

print(chain.invoke(messages))