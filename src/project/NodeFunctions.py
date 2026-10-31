from project.Prompts import (
    LTM_Query_Rewriting_Prompt,
    answer_prompt,
    memory_extraction_prompt,
    router_prompt,
)
from project.models import llm_with_websearch, brain, ltm_model, memory_model
from project.classes import STATE
from langchain_tavily import TavilySearch
from project.ltm import retrieve_memories, save_memory
from dotenv import load_dotenv
load_dotenv()
from langchain_core.messages import HumanMessage,AIMessage

# save newly generated long term memory in database
def saveFunction(state: STATE) -> STATE:
    conversation = state.get("conversation", [])
    if len(conversation) < 2:
        return {"generated_memory": []}

    exchange = "\n".join(
        f"{message.type}: {message.content}"
        for message in conversation[-2:]
    )
    response = memory_model.invoke(
        memory_extraction_prompt.invoke({"exchange": exchange})
    )

    for memory in response.memories:
        memory = memory.strip()
        if memory:
            save_memory(memory)

    return {"generated_memory": response.memories}

# retrieve already present long term memory from database
def retrieveFunc(state: STATE) -> STATE:
    prompt_ = LTM_Query_Rewriting_Prompt.invoke(
        {
            "chat_history": state.get("conversation", []),
            "user_query": state['input_query']
        }
    )

    response = ltm_model.invoke(prompt_)

    query=response.ltm_query

    memories = retrieve_memories(query)

    return {
        "retrieved_memory": "\n".join(memories)
    }








# function to decide websearch
def router_func(state: STATE) -> STATE:
    response = llm_with_websearch.invoke(
        router_prompt.invoke({
            "chat_history": state.get("conversation", []),
            "current_query": state["input_query"]
        })
    )
    return {
        "websearch_": response.websearch_,
        "modified_query": response.modified_query
    }


# function to websearch
def websearch_func(state: STATE) -> STATE:
    search = TavilySearch(
        max_results=5
    )
    search_response = search.invoke({
        "query": state["modified_query"]
    })
    documents = [
        item["content"]
        for item in search_response["results"]
    ]
    return {
        "results": documents
    }



# Main function
def chat_func(state: STATE) -> STATE:
    results = "\n\n".join(
        state.get("results", [])
    )

    messages = answer_prompt.invoke({

        "current_query": state["input_query"],

        "chat_history": state.get(
            "conversation",
            []
        ),

        "results": results,
        "retrieved_memory": state.get("retrieved_memory", "")
    })

    response = brain.invoke(messages)

    return {
        "conversation": [
            HumanMessage(
                content=state["input_query"]
            ),

            AIMessage(
                content=response.content
            )
        ]
    }

# condition_func
def condition_func(state: STATE):

    if state["websearch_"] == "YES":
        return "websearch_Node"

    return "chat_Node"
