from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import create_react_agent

load_dotenv()

model = ChatAnthropic(
    model="claude-sonnet-4-5",
    temperature=0.1
)

memory = SqliteSaver.from_conn_string(":memory:")

tools = [
    TavilySearchResults(max_results=2)
]

agent_executor = create_react_agent(
    model,
    tools,
    checkpointer=memory
)

config = {
    "configurable": {
        "thread_id": "abc123"
    }
}

while True:
    user_input = input("Sen: ").strip()

    response = agent_executor.invoke(
        {
            "messages": [
                HumanMessage(
                    content=f"Kullanıcıya yalnızca Türkçe cevap ver. Kullanıcının mesajı: {user_input}"
                )
            ]
        },
        config=config
    )

    print("Agent:", response["messages"][-1].content)