"""
this meory with out to with rh,cmsh, mph
"""

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder

 
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# load the .env
load_dotenv()

# create the chat model
model=ChatOpenAI(model="gpt-4o-mini")

# First call
firstresponse = model.invoke(
    [HumanMessage(content="Hi! I'm uday")]
)
print(firstresponse.content)


# second call
secondresponse=model.invoke(
    [HumanMessage(content="what's my name?")]
)
print(secondresponse.content)

# third call
thirdresponse=model.invoke(
    [HumanMessage(content="Hi! I'm uday"),
    AIMessage(content="Hi Uday! How can I assist you today?"),
    HumanMessage(content="what's my name?")]
)
print(thirdresponse.content)


# Now create one prompt template using Messageplaceholder
prompt_template=ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "you are a helpful assistant. Answer all questons to the best of your ability.",
            ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# connect the prompt and model
chain= prompt_template | model

# Excute the chain with convertion history
response= chain.invoke(
    {
        "messages":[
            HumanMessage(content="Hi! I'm uday"),
    AIMessage(content="Hi Uday! How can I assist you today?"),
    HumanMessage(content="what's my name?")
        ]
    }
) 
print(response.content)




# now create the lanchain memory with ChatMessageHistory and RunnableWithMessageHistory 
# Create the prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer questions clearly.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


# Connect the prompt and model
chain = prompt | model


# Create a store for different sessions
store = {}


#Get or create history for a session
def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()

    return store[session_id]


#Add automatic message history to the chain
model_with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages",
)


#Start a chat session
session_id = "1"

while True:
    content = input("Enter your message (type 'exit' to stop): ")

    if content.lower() == "exit":
        break

    config = {
        "configurable": {
            "session_id": session_id
        }
    }

    response = model_with_message_history.invoke(
        {"messages": [HumanMessage(content=content)]},
        config=config,
    )

    print("AI:", response.content)


#Display the conversation history
print("\nConversation history:")

print(
    model_with_message_history
    .get_session_history(session_id)
    .messages
)

print(store)


