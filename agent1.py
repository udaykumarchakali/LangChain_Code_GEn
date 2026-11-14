
from dotenv import load_dotenv
import os

load_dotenv()

from typing import Union, List

from pydantic import BaseModel, Field

from langchain.tools import tool
from langchain_core.tools import StructuredTool, Tool
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.tools.render import render_text_description

from langchain_openai import ChatOpenAI

from langchain_community.tools.tavily_search import TavilySearchResults

from langchain_classic.agents.output_parsers import ReActSingleInputOutputParser
from langchain_classic.agents.format_scratchpad import format_log_to_str


 
# Creating tools using @tool
 

@tool
def add(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b


@tool
def subtract(a: int, b: int) -> int:
    """Subtracts b from a."""
    return a - b


@tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b."""
    return a * b


print(add.invoke({"a": 10, "b": 20}))
print(subtract.invoke({"a": 10, "b": 20}))
print(multiply.invoke({"a": 10, "b": 20}))


 
# Creating a tool using Pydantic BaseModel
 

class SearchInput(BaseModel):

    query: str = Field(
        description="Should be a search query"
    )


@tool("search-tool", args_schema=SearchInput)
def search(query: str) -> str:
    """Look up things online."""
    return "LangChain"


result4 = search.invoke({
    "query": "What is LangChain?"
})

print(result4)

 
# Creating a tool using StructuredTool.from_function()
 

def mymultiply(a: int, b: int) -> int:
    """Multiplies a and b."""
    return a * b


class MultiplyInput(BaseModel):

    a: int = Field(..., description="First integer")
    b: int = Field(..., description="Second integer")


mytool = StructuredTool.from_function(
    func=mymultiply,
    description="Multiply 2 numbers",
    args_schema=MultiplyInput,
    name="mmmultiply"
)

print(mytool.invoke({"a": 10, "b": 5}))


 
# Tavily Search Tool
 

tavilysearch = TavilySearchResults(max_results=5)


 
# Connecting tools with LLM using bind_tools()
 

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

tools = [
    tavilysearch,
    add,
    subtract,
    multiply
]

llm_with_tools = llm.bind_tools(tools)

messages = [
    HumanMessage(content="Tell me about Virat Kohli")
]

response = llm_with_tools.invoke(messages)

print("Content:", response.content)
print("Tool Calls:", response.tool_calls)

 
# Execute Tool Calling
 

if response.tool_calls:

    messages.append(response)

    for tool_call in response.tool_calls:

        tool_name = tool_call["name"]

        tool_args = tool_call["args"]

        if tool_name == tavilysearch.name:
            tool_result = tavilysearch.invoke(tool_args)

        elif tool_name == add.name:
            tool_result = add.invoke(tool_args)

        elif tool_name == subtract.name:
            tool_result = subtract.invoke(tool_args)

        elif tool_name == multiply.name:
            tool_result = multiply.invoke(tool_args)

        else:
            raise ValueError(f"Tool {tool_name} not found")

        print("Tool Name:", tool_name)
        print("Tool Result:", tool_result)

        messages.append(
            ToolMessage(
                content=str(tool_result),
                tool_call_id=tool_call["id"]
            )
        )

    final_response = llm_with_tools.invoke(messages)

    print("\nFinal Answer:")
    print(final_response.content)

else:
    print("\nNormal LLM Response:")
    print(response.content)


 
# ReAct Agent
 

print("\n********************** ReAct **************************")


# Tools for ReAct Agent
react_tools: List[Tool] = [
    add,
    subtract,
    multiply,
    tavilysearch
]


# Find a particular tool using its name
def find_tool_by_name(
    tools: List[Tool],
    tool_name: str
) -> Tool:

    for tool_item in tools:

        if tool_item.name == tool_name:
            return tool_item

    raise ValueError(f"Tool with name {tool_name} not found")


# Create a ReAct Prompt Template
template = """
Answer the following questions as best you can.

You have access to the following tools:

{tools}

You must use these tools to answer the question.

Use the following format:

Question: the input question you must answer

Thought: think about what to do

Action: the action to take, should be one of [{tool_names}]

Action Input: the input to the action

Observation: the result of the action

...(this Thought/Action/Action Input/Observation
can repeat N times)

Thought: I now know the final answer

Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""


# Create PromptTemplate
prompt = PromptTemplate.from_template(
    template=template
).partial(
    tool_names=", ".join([t.name for t in react_tools]),
    tools=render_text_description(react_tools)
)


# Create LLM for ReAct
react_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
).bind(
    stop=["\nObservation"]
)


# Create intermediate steps
intermediate_steps = []


# Create ReAct Agent Chain
agent = (
    {
        "input": lambda x: x["input"],

        "agent_scratchpad": lambda x: format_log_to_str(
            x["intermediate_steps"]
        )
    }
    | prompt
    | react_llm
    | ReActSingleInputOutputParser()
)


# Run the ReAct Agent
agent_step: Union[AgentAction, AgentFinish] = None

question = "Tell me about Uday Kumar Chakali"

for i in range(10):

    agent_step = agent.invoke({
        "input": question,
        "intermediate_steps": intermediate_steps
    })

    print("\nAgent Step:", agent_step)

    if isinstance(agent_step, AgentAction):

        tool_name = agent_step.tool

        tool_to_use = find_tool_by_name(
            react_tools,
            tool_name
        )

        tool_input = agent_step.tool_input

        observation = tool_to_use.invoke(tool_input)

        print("Tool Name:", tool_name)
        print("Tool Input:", tool_input)
        print("Observation:", observation)

        intermediate_steps.append(
            (agent_step, str(observation))
        )

    elif isinstance(agent_step, AgentFinish):

        print("\nFinal Answer:")
        print(agent_step.return_values["output"])

        break

else:
    print("Maximum iterations reached.")



# What is 2+3 multiplied by 4

intermediate_steps = []

question = "What is 2+3 multiplied by 4"

print("Input:", question)

agent_step: Union[AgentAction, AgentFinish] = None

for i in range(10):

    agent_step = agent.invoke({
        "input": question,
        "intermediate_steps": intermediate_steps
    })

    # Check AgentAction
    if isinstance(agent_step, AgentAction):

        print("\nThought:", agent_step.log.split("Action:")[0])
        print("Action:", agent_step.tool)

        tool_name = agent_step.tool
        tool_input = agent_step.tool_input

        print("Action Input:", tool_input)

        tool_to_use = find_tool_by_name(
            react_tools,
            tool_name
        )

        # Execute calculator tools
        if tool_name in ["add", "subtract", "multiply"]:

            numbers = [
                int(x.strip())
                for x in tool_input.split(",")
            ]

            observation = tool_to_use.invoke({
                "a": numbers[0],
                "b": numbers[1]
            })

        else:
            observation = tool_to_use.invoke(tool_input)

        print("Observation:", observation)

        intermediate_steps.append(
            (agent_step, str(observation))
        )

    # Check AgentFinish
    elif isinstance(agent_step, AgentFinish):

        print("\nAgentFinish:")
        print("Final Answer:", agent_step.return_values["output"])

        break

else:
    print("Maximum iterations reached.")
    


print("---------------------------------Create_Agent_in_Short_using_Langsmith-----------------------------------------------")

from dotenv import load_dotenv
from langsmith import Client
from langchain_classic.agents import create_react_agent, AgentExecutor
load_dotenv
 
# Get the ReAct Prompt from langsmith using Client.pull_prompt 
client = Client()
prompt = client.pull_prompt(
    "hwchase17/react",
    dangerously_pull_public_prompt=True
)
print(prompt)

tools = [tavilysearch,add,subtract,multiply]
llm=ChatOpenAI(
    model="gpt-4o-mini",
    stop_sequences=["\nObservation","Observation"]
)

agent=create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)
agent_executor=AgentExecutor(agent=agent,tools=tools,verbose=True,handle_parse_errors=True)
uday=agent_executor.invoke({"input":"Tell me about virat in 5 lines"})




print("---------------------------------Create_tool_calling_Agent_in_Short_using_Langsmith-----------------------------------------------")

from langchain_classic.agents import create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate,HumanMessagePromptTemplate,MessagesPlaceholder
from langchain_classic.schema import SystemMessage

tools=[tavilysearch,add,subtract,multiply]

llm=ChatOpenAI(
    model="gpt-4o-mini",
    # stop_sequences=["\nObservation","Observation"]
)

prompt=ChatPromptTemplate(
    messages=[
        SystemMessage(content=(
            "You are an AI."
            "You should not use tools parallelly"
            "You should not make your own conclusions even if it is easy. instead use appropriate tools"
        )),
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ]
)
agent=create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor=AgentExecutor(agent=agent, tools=tools,verbose=True,handle_parsing_errors=True)

uday1=agent_executor.invoke({"input":"Tell me about Dhoni"})
print(uday1)



print("---------------------------------Wikipedia-----------------------------------------------")

import requests

# Get user question
query = input("Enter your question: ")

# Wikipedia API URL
url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + query.replace(" ", "_")

headers = {
    "User-Agent": "UdayLearningProject/1.0"
}

# Send request
response = requests.get(url, headers=headers)

# Get answer
if response.status_code == 200:
    data = response.json()

    print("\nAnswer:")
    print(data["extract"])

else:
    print("Sorry, information not found!")




print("---------------------------------Google_Search-----------------------------------------------")

from langchain_community.utilities import SerpAPIWrapper
from dotenv import load_dotenv

load_dotenv()

serp=SerpAPIWrapper()
prasad=serp.run("Ram Charan")
print(prasad)










































































































































































































































































































































































































































































































