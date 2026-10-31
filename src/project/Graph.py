from langgraph.graph import StateGraph, START, END
from project.classes import STATE
from project.NodeFunctions import *


graph = StateGraph(STATE)

graph.add_node(
    "router_Node",
    router_func
)
graph.add_node(
    "websearch_Node",
    websearch_func
)

graph.add_node(
    "chat_Node",
    chat_func
)

graph.add_node("retrieve_node",retrieveFunc)

graph.add_node("save_node",saveFunction)

graph.add_edge(
    START,
    "retrieve_node"
)

graph.add_edge("retrieve_node", "router_Node")


graph.add_conditional_edges(
    "router_Node",
    condition_func,
    {
        "websearch_Node": "websearch_Node",
        "chat_Node": "chat_Node"
    }
)


graph.add_edge(
    "websearch_Node",
    "chat_Node"
)

graph.add_edge(
    "chat_Node",
    "save_node"
)

graph.add_edge(
    "save_node",
    END
)
