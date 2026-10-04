#This file serves as the foundation of the project by defining the capabilities the AI agent 
# will have access to.It loads environment variables, which is necessary for authenticating 
# the API keys required for Google Gemini and Tavily.   It defines a custom Python function 
# triple and decorates it with @tool, turning it into a tool the LLM can natively understand
#  and call to multiply a given number by 3.   It configures a list of available tools, 
# pairing the custom triple tool with TavilySearch, which is configured to return a maximum 
# of one search result.   It initializes the LLM using ChatGoogleGenerativeAI with the 
# gemini-3.8-flash model and a temperature of 0 for deterministic, focused responses.   
# Crucially, it uses .bind_tools(tools) to inform the LLM about the available tools and 
# their expected input schemas.  

from dotenv import load_dotenv
from langchain_core.tools import tool
#from langchain_openai import ChatOpenAI
#from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch

load_dotenv()

@tool
def triple(num:float) -> float:
    """
    param num: a number to triple
    returns: the triple of the input number
    """
    return float(num) * 3

tools = [TavilySearch(max_results=1), triple]

#llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0).bind_tools(tools)

# Initializing ChatOllama with one of your locally installed models
llm = ChatOllama(model="llama3.2:3b", temperature=0).bind_tools(tools)
