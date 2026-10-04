#This LangGraph project implements a ReAct (Reasoning and Acting) agent capable of 
#answering questions by deciding when to search the web and when to perform custom calculations. 

#This is the main execution script where the individual components are wired together into 
# a cyclical graph structure.It defines a routing function named should_continue that checks 
# the most recent message in the graph's state (state["messages"][-1]). If the LLM generated 
# tool_calls, it returns the string "act"; otherwise, it returns END to terminate the loop.  
#  It initializes a StateGraph using MessagesState, which acts as the memory passing between nodes.  
#  It adds the two core nodes to the graph (agent_reason and act) and designates agent_reason 
# as the starting entry point.   It establishes the workflow logic by adding a conditional edge
# from agent_reason (using the should_continue logic) and a standard edge from act back to agent_reason. 
# This creates the classic ReAct loop: the agent reasons, decides to act (use a tool), the tool executes, 
# and the result goes back to the agent to reason again until it has the final answer.  
#  It compiles the defined flow into an executable app and generates a visual Mermaid diagram (flow.png) 
# of the architecture.   
# Finally, it tests the graph by invoking it with a complex prompt: "What is the temperature in Pune? 
# List it and then triple it". This specific prompt forces the graph to loop multiple times, 
# as the LLM must first call Tavily to find the temperature, ingest the result, and then call
#  the triple tool on that number before outputting the final message.   "


from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState, StateGraph,END

from nodes import run_agent_reasoning, tool_node

load_dotenv()

AGENT_REASON="agent_reason"
ACT= "act"
LAST = -1


def should_continue(state: MessagesState) -> str:
    if not state["messages"][LAST].tool_calls:
        return END
    return ACT

flow = StateGraph(MessagesState)

flow.add_node(AGENT_REASON, run_agent_reasoning)
flow.set_entry_point(AGENT_REASON)
flow.add_node(ACT, tool_node)

flow.add_conditional_edges(AGENT_REASON, should_continue, {
    END:END,
    ACT:ACT})

flow.add_edge(ACT, AGENT_REASON)

app = flow.compile()
app.get_graph().draw_mermaid_png(output_file_path="flow.png")

if __name__ == "__main__":
    print("Hello ReAct LangGraph with Function Calling")
    res = app.invoke({"messages": [HumanMessage(content="What is the temperature in Pune? List it and then triple it")]})
    print(res["messages"][LAST].content)