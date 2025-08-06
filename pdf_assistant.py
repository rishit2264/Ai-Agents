import  typer         #Typer is a Python library that helps you create command-line interfaces (CLIs).  it adds modern Python features like type hints to make development easier and more robust.
from typing import Optional,List
from phi.assistant import Assistant   #independent autonomous AI which will assist with our tasks
from  phi.storage.assistant.postgres import PgAssistantStorage      #will create a session  This particular class is used to store and retrieve assistant-related data (like conversation history, assistant configurations, etc.) in a PostgreSQL database.
from phi.knowledge.pdf import PDFUrlKnowledgeBase     #to interact with the pdf uploaded
from phi.vectordb.pgvector import PgVector2    #it is basically a good vector database integration for postgresSQL providing good featuress such as knn search ,vector similarity etc.


import os
from dotenv import load_dotenv    #loads things from .env file
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Remove OpenAI key to force using Groq
if "OPENAI_API_KEY" in os.environ:
    del os.environ["OPENAI_API_KEY"]

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"        #locally in docker

knowledge_base = PDFUrlKnowledgeBase(
    urls = ["https://www.researchgate.net/publication/330533824_TRADITIONAL_FOODS_OF_INDIA"],       #location of pdf
    vector_db=PgVector2(collection = "recipes",db_url = db_url)
)

knowledge_base.load()

storage = PgAssistantStorage(table_name="pdf_assistant",db_url=db_url)  

def pdf_assistant(new : bool = False ,user:str = "user"):         #name of function should be same as table_name above  in storage
    run_id : Optional[str] = None            #run id check if session in docker was running previously and extracts the data from it.

    if not new:
        existing_run_ids : List[str] = storage.get_all_run_ids(user)
        if len(existing_run_ids) >0:
            run_id = existing_run_ids[0]

    assistant = Assistant(
        run_id = run_id,
        user_id = user,
        knowledge_base =knowledge_base,
        storage = storage,
        show_tool_calls=True,       #shows the tool calls
        search_knowledge=True,      #enables agent to search knowledge base
        read_chat_history=True,     #enable assistant to read chat history
    )

    if run_id is None:
        run_id = assistant.run_id
        print(f"Started run:{run_id}\n")
    else:
        print(f"continuing run:{run_id}\n")

    assistant.cli_app(markdown=True)         #running in a command line prompt

if __name__ == "__main__":
    typer.run(pdf_assistant)