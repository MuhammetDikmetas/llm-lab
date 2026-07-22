from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent



load_dotenv()

model = ChatAnthropic(
    model="claude-sonnet-4-5",
    temperature=0.1
)


search=TavilySearchResults(max_results=2)

tools=[search]



agent_executor=create_react_agent(model,tools)



response=agent_executor.invoke(
    {"messages":[HumanMessage(content="What is the weather in Istanbul now?")]}
)


print(response["messages"][-1].content)


