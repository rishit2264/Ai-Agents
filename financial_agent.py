from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools
#import openai

#import os
#from dotenv import load_dotenv
#load_dotenv()
#openai.api_key = os.getenv("OPENAI_API_KEY")


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


#now to combine both we make a multi ai agent

# multi agent application
multi_ai_agent=Agent(
    team=[web_search_agent,finance_agent],
    model=Groq(id="llama3-70b-8192"),
    instructions=["Always include sources", "Use tables to display data"],
    show_tool_calls=True,
    markdown=True,
)

#now to work it out:
#multi_ai_agent.print_response("summarize analyst recommendation and share the latest news for NVDA",stream = True)
#finance_agent.print_response("Get analyst recommendations for NVDA", stream=True)
multi_ai_agent.print_response("Get the analyst recommendations for NVDA", stream=True)
multi_ai_agent.print_response("What is the latest news about NVDA?", stream=True)


