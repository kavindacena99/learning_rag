from typing import TypedDict
from langgraph.graph import StateGraph, START, END

#class State(TypedDict):
#    message: str

# second learn about state
#class State(TypedDict):
    #name: str
    #age: str
    #answer: str

class RAGState(TypedDict):
    question: str
    answer: str
    document: list[str]

"""
def node_greet(state: State):
    return {
        "name": "Hello " + state["name"]
    }

def node_get_name(state: State):
    return {
        "age": " You are " + state["age"] + " years old."
    }

"""

def retrieve(state: RAGState):
    question = state["question"]

    document = [
        "I am Kavinda, a software engineer.",
        "I have experience in Python, JavaScript, and cloud technologies.",
        "Now I am working on Langgraph"
    ]

    return {
        "document": document
    }

def generate(state: RAGState):
    document = state["document"]
    question = state["question"]

    answer = f""" Based on docs: {document} Answer to: {question} """

    return {
        "answer": answer
    }


graph = StateGraph(RAGState)

graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)

graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)

app = graph.compile()

result = app.invoke({
    "question": "What is your experience?",
    "document": []  
})

print(result) # Output: {'message': 'Hi Hello World'}