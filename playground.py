from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools
import openai
from phi.model.openai import OpenAIChat
import phi
from phi.playground import Playground,serve_playground_app

import os
from dotenv import load_dotenv
load_dotenv()

phi.api = os.getenv("PHI_API_KEY")

#web search agent
web_search_agent = Agent(
    name = "Web_search_agent",
    role = "search the web  for the information",  #Defines the agent's role/purpose, which is to search the web for information.
    model = Groq(id = "llama3-70b-8192"),  
    tools = [DuckDuckGo()],                        #Provides the agent with access to the DuckDuckGo search tool, which it can use to find information on the web.
    instruction = ["always include sources"],      #Gives a specific instruction to the agent to always cite sources when providing information.
    show_tool_calls=True,                          # Enables visibility of tool calls, meaning the agent will show when and how it's using tools like DuckDuckGo in its responses.
    markdown=True                                  #Configures the agent to format its responses using Markdown, which allows for better text formatting with headings, lists, links, etc.
)

#Financial agent

finance_agent = Agent(
    name = "Finance AI agent",
    model = Groq(id = "llama3-70b-8192"),
    tools = [
        YFinanceTools(stock_price = True,analyst_recommendations=True,stock_fundamentals=True,company_news=True)   #has the data regarding the stocks and finance
    ],
    instructions=["Use tables to dispay the data"],
    show_tool_calls=True,
    markdown=True

)

app = Playground(agents = [finance_agent,web_search_agent]).get_app()

if __name__ == "__main__":
    serve_playground_app("playground:app",reload = True)
