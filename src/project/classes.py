from  typing import TypedDict, Literal, Annotated
from langchain_core.messages import BaseMessage
from pydantic import BaseModel , Field
from langgraph.graph.message import add_messages


# 1 workflow State
class STATE(TypedDict):
    conversation: Annotated[list[BaseMessage], add_messages]
    retrieved_memory: str
    input_query: str
    websearch_: Literal["YES", "NO"]
    results: list[str]
    modified_query: str
    generated_memory : list[str]


# 2. class for structured output

class IsWebSearchRequired(BaseModel):
    websearch_: Literal["YES", "NO"] = Field(
        ...,
        description="Whether an internet search is required to answer the user's request."
    )
    modified_query: str = Field(
        ...,
        description="A concise and specific query to use for web search or answering the user's request."
    )

class queryForltm(BaseModel):
    ltm_query : str = Field(
        ...,
        description='generate query to retrieve relevent informations for ltm '
    )


class MemoriesToSave(BaseModel):
    memories: list[str] = Field(
        default_factory=list,
        description=(
            "A list of concise, durable facts about the user that are worth "
            "remembering across conversations. Return an empty list when there "
            "is nothing stable or useful to save."
        ),
    )

