"""A small planner / worker / reviewer graph (fixture: scanned, never run)."""
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from app.prompts import REVIEWER_PROMPT

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, max_tokens=512)


@tool
def write_file(path: str, text: str) -> str:
    """Write text to a file in the workspace and return the path."""
    return path


def planner(state):
    msgs = [
        SystemMessage(content="You are the planner. Break the task into steps."),
        HumanMessage(state["task"]),
    ]
    return llm.invoke(msgs)


def worker(state):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a careful worker. "
                   'Use the "write_file" tool.'),
        ("human", "{step}"),
    ])
    return prompt


def reviewer(state):
    return [SystemMessage(content=REVIEWER_PROMPT),
            HumanMessage(content=f"Review step {state['n']} strictly.")]


def build(graph):
    return graph.compile().with_config({"recursion_limit": 25})
