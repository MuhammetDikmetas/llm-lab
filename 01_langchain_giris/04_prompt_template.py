from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

model = ChatAnthropic(
    model="claude-sonnet-4-5",
    temperature=0.1
)

system_prompt = "{city} için {days} günlük kısa bir gezi planı hazırla."

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", "Gezi planını oluştur.")
])

parser = StrOutputParser()

chain = prompt_template | model | parser

print(chain.invoke({
    "city": "İstanbul",
    "days": 2
}))