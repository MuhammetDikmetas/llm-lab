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
        content="Sen bir çeviri asistanısın. Verilen Türkçe metni İngilizceye çevir."
    ),
    HumanMessage(
        content="Yapay zekâ öğreniyorum."
    )
]


parser = StrOutputParser()

response = model.invoke(messages)

print(parser.invoke(response))

