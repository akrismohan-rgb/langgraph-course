#This file defines the distinct steps, or "nodes," that the LangGraph state machine will transition 
#between during execution.It imports the pre-configured llm and tools list from react.py.   
# It establishes a SYSYEM_MESSAGE that instructs the LLM on its persona as a helpful assistant
#  capable of using tools.   It defines the run_agent_reasoning function, which serves as the
#  core decision-making node. This function takes the current conversation state (MessagesState), 
# formats a prompt containing the system message and the history of user/agent messages, and 
# invokes the LLM. The LLM's response is then returned to be appended to the state.   It creates a 
# tool_node utilizing LangGraph's prebuilt ToolNode class. This node is initialized with the 
# project's tools and is responsible for actually executing the Python functions when the LLM 
# requests a tool call.  

from dotenv import load_dotenv
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode

from react import llm, tools

load_dotenv()

SYSYEM_MESSAGE="""
You are a helpful assistant that can use tools to answer questions.
"""

def run_agent_reasoning(state: MessagesState) -> MessagesState:
    """
    Run the agent reasoning node.
    """
    response = llm.invoke([{"role": "system", "content": SYSYEM_MESSAGE}, *state["messages"]])
    return {"messages": [response]}

tool_node = ToolNode(tools)