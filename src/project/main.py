import os
from dotenv import load_dotenv
from project.Graph import graph
from langgraph.checkpoint.postgres import PostgresSaver
from project.ltm import create_memory_table

load_dotenv()


DB_URI=os.getenv("POSTGRES_URI")

create_memory_table()

thread_id=input("Enter the thread ID corresponding to the conversation you want: ")


with PostgresSaver.from_conn_string(DB_URI) as checkpoint:

    checkpoint.setup()
    # creates the tables required by LangGraph in PostgreSQL.

    app = graph.compile(checkpointer=checkpoint)
    config = {
        "configurable":{
            "thread_id": thread_id,
        }
    }

    print(
        app.get_graph().draw_ascii()
    )

    while True:
        user_input = input("Enter your message (or 'exit' to quit): ")

        if user_input.lower() == "exit":
            break

        state = {
            "input_query": user_input,
            "conversation": [],
            "websearch_": "",
            "results": [],
            "modified_query": "",
            "retrieved_memory": "",
            "generated_memory": []
        }

        final_state = app.invoke(state, config=config)

        print("Aria:", final_state["conversation"][-1].content)

