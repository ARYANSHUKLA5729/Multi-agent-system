from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from tools import web_search, scrape_url

llm = ChatGroq(
    model="llama-3.1-8b-instant",  # or another supported model (e.g., mixtral-8x7b-32768)
    temperature=0.2
)

## creating 1 st agent

def build_search_agent():
    return create_agent(
        model=llm,
        tools=[web_search]
    )
    
## creating 2 agent

def builder_reader_agent():
    return create_agent(
        model=llm,
        tools=[scrape_url]
    )
    
## writing agent

writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Write clear, structured and insightful report"),
    ("human", """Write a detailed research report on the topic below

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list all URLs found in the research)

Be detailed, factual and professional.""")
])

writer_chain = writer_prompt | llm | StrOutputParser()

# critic chain

critic_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a sharp and constructive research critic. Be honest and specific."),
    ("human", """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:

Areas to Improve:

-

TER

One line verdict:
...""")
])

critic_chain = critic_prompt | llm | StrOutputParser()