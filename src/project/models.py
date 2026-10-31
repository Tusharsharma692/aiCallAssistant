from langchain_openrouter import ChatOpenRouter
from dotenv import load_dotenv
from project.classes import IsWebSearchRequired, MemoriesToSave, queryForltm

load_dotenv()

brain = ChatOpenRouter(
    model="anthropic/claude-sonnet-4-5",
    temperature=0,
    max_tokens=200
)

chat_model = ChatOpenRouter(
    model="anthropic/claude-sonnet-4-5",
    temperature=0,
    max_tokens=300
)



llm_with_websearch = chat_model.with_structured_output(
    IsWebSearchRequired
)

ltm_query = ChatOpenRouter(
    model="anthropic/claude-sonnet-4-5",
    temperature=0.5,
    max_tokens=150
)

ltm_model=ltm_query.with_structured_output(
    queryForltm
)

memory_model = chat_model.with_structured_output(MemoriesToSave)

