
"""
memory converstion history with gradio
"""

import gradio as gr
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# load the .env
load_dotenv()

# create the chat model
model=ChatOpenAI(model="gpt-4o-mini")

# Store histories for different sessions
store = {}


# Get or create session history
def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# Add message history to the model
model_with_message_history = RunnableWithMessageHistory(
    model,
    get_session_history,
)


# Gradio chatbot function
def predict(message, history, request: gr.Request):
    session_id = request.session_hash

    config = {
        "configurable": {
            "session_id": session_id
        }
    }

    response = model_with_message_history.invoke(
        [HumanMessage(content=message)],
        config=config,
    )

    return response.content


# Launch Gradio chatbot
gr.ChatInterface(
    fn=predict,
    title="Student AI Study Assistant",
).launch(debug=True)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
        ),
        MessagesPlaceholder(variable_name="mychat_history"),
        ("human", "{input}")
    ]
)

chain = prompt | model

model_with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    history_messages_key="mychat_history",
    input_messages_key="myinput"
)

config = {"configurable": {"session_id": "222"}}

content = input(">> ")

result = model_with_message_history.invoke(
    {"myinput": content, "language": "english"},
    config=config,
)

result




# show the conversation history 

# from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain_community.chat_message_histories import ChatMessageHistory

# load_dotenv()

# model = ChatOpenAI(model="gpt-4o-mini")

# # Store conversation histories
# store = {}

# def get_session_history(session_id: str):
#     if session_id not in store:
#         store[session_id] = ChatMessageHistory()
#     return store[session_id]

# # Create prompt
# prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             "You are a helpful assistant. Answer all questions "
#             "to the best of your ability in {language}.",
#         ),
#         MessagesPlaceholder(variable_name="mychat_history"),
#         ("human", "{myinput}"),
#     ]
# )

# chain = prompt | model

# # Add message history
# model_with_message_history = RunnableWithMessageHistory(
#     chain,
#     get_session_history,
#     history_messages_key="mychat_history",
#     input_messages_key="myinput",
# )

# config = {"configurable": {"session_id": "333"}}

# while True:
#     content = input(">> ")

#     if content.lower() == "exit":
#         break

#     result = model_with_message_history.invoke(
#         {"myinput": content, "language": "English"},
#         config=config,
#     )

#     print("AI:", result.content)
    
# #print the current conversation history id 
# print("All Session IDs:", list(store.keys()))

# # session id and get the history
# session_id=input("enter the Session Id:")
# if session_id in store:
#     history= get_session_history(session_id)
    
#     print(f"\nHistory for Session ID:{session_id}")
    
#     for message in history.messages:
#         print(f"{message.type}:{message.content}")
# else:print("session Id not found!")