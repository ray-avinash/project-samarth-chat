# File: backend/agent.py
# --- FINAL "SLIM-BUT-SMART" VERSION ---

# Import necessary libraries
import pandas as pd
import sqlite3
import ast
import os
import re # <-- Added import
from typing import TypedDict, List, Dict, Optional
from sqlalchemy import create_engine
from dotenv import load_dotenv

# LangChain and LangGraph specific imports
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import Tool

from langgraph.graph import StateGraph, END

# --- CRITICAL CONFIGURATION ---
load_dotenv() 
groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable not set. Please set it before running the app.")


# --- DATABASE CONFIGURATION --
DATABASE_URL = os.environ.get("DATABASE_URL") 

# This fix is needed for SQLAlchemy to work with Railway's URL
if DATABASE_URL and DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://", 1)

print("Connecting to production database...")
engine = create_engine(DATABASE_URL)

# Define exactly which tables the agent can access
include_these_tables = ['crop_production', 'rainfall']

db = SQLDatabase(
    engine,
    include_tables=include_these_tables,
    sample_rows_in_table_info=0  # Keep the previous fix too
)

# Use the new (non-deprecated) function to get table names
print(f"Database connection successful. Agent is restricted to tables: {db.get_usable_table_names()}")

# Define AgentState
class AgentState(TypedDict):
    user_query: str
    rewritten_query: str
    chat_history: List[Dict]
    collected_data: List[Dict]
    planner_decision: Optional[Dict]
    final_response: Optional[str]


# Initialize the language model
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, groq_api_key=groq_api_key)

# Define helper functions for the SQL Chain
def run_sql_query(query: str):
    return db.run(query)

def parse_sql_result(result_str: str) -> str:
    try:
        data = ast.literal_eval(result_str)
        if isinstance(data, list) and len(data) == 1 and isinstance(data[0], tuple) and len(data[0]) == 1:
            return str(data[0][0])
        if isinstance(data, list) and len(data) == 1 and isinstance(data[0], tuple) and len(data[0]) > 1:
            return ", ".join([str(item) for item in data[0]])
        if isinstance(data, list) and all(isinstance(t, tuple) for t in data):
             clean_tuples = []
             for t in data:
                 if len(t) == 1:
                     clean_tuples.append(f"{t[0]}")
                 elif len(t) == 2:
                     clean_tuples.append(f"{t[0]} ({t[1]})")
             return ", ".join(clean_tuples)
        return result_str
    except Exception:
        return result_str

DB_SCHEMA = """
CREATE TABLE crop_production (
  state_name TEXT,
  district_name TEXT,
  crop_year INTEGER,
  season TEXT,
  crop TEXT,
  area_hectares REAL,
  production_tonnes REAL
)

/*
3 rows from crop_production table:
state_name  district_name crop_year season  crop  area_hectares production_tonnes
Andaman and Nicobar Islands NICOBARS  2000  Kharif  Arecanut  1254.0  2000.0
Andaman and Nicobar Islands NICOBARS  2000  Kharif  Other Kharif Pulses 2.0 1.0
Andaman and Nicobar Islands NICOBARS  2000  Kharif  Rice (Paddy)  102.0 321.0
*/


CREATE TABLE rainfall (
  "SUBDIVISION" TEXT,
  "YEAR" INTEGER,
  "JAN" REAL,
  "FEB" REAL,
  "MAR" REAL,
  "APR" REAL,
  "MAY" REAL,
  "JUN" REAL,
  "JUL" REAL,
  "AUG" REAL,
  "SEP" REAL,
  "OCT" REAL,
  "NOV" REAL,
  "DEC" REAL,
  "ANNUAL" REAL,
  "WINTER" REAL,
  "PRE_MONSOON" REAL,
  "MONSOON" REAL,
  "POST_MONSOON" REAL
)

/*
3 rows from rainfall table:
SUBDIVISION YEAR  JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC ANNUAL  WINTER  PRE_MONSOON MONSOON POST_MONSOON
Andaman & Nicobar Islands 1997  9.5 0.0 0.2 15.6  281.1 199.5 918.5 430.6 440.2 128.7 292.8 38.4  2755.1  9.5 296.9 1988.8  459.9
Andaman & Nicobar Islands 1998  0.9   0.0 0.0 0.0 348.9 600.0 364.5 258.9 337.8 618.6 227.8 89.0  2846.4  0.9 348.9   1561.2  935.4
Andaman & Nicobar Islands 1999  46.8  44.6  14.2  270.6 257.4 295.0 408.5 329.2 325.3 437.5 124.9 145.7 2699.7  91.4  542.2   1358.0  708.1
*/
"""

sql_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a SQLite expert. Given a user question, you must generate a
            syntactically correct SQLite query to answer it.

            **DB Schema:**
            {schema}

            **CRITICAL SCHEMA NOTES:**
            - 'rainfall' table: 'SUBDIVISION' is state, 'ANNUAL' is total rainfall.
            - 'crop_production' table: 'state_name' is state, 'crop' is crop name, 'production_tonnes' is volume.

            **CRITICAL LOGIC NOTES:**
            1.  **FOR "LAST N YEARS":**
                - Use Subqueries such as `WHERE crop_year IN (SELECT DISTINCT crop_year FROM crop_production ORDER BY crop_year DESC LIMIT N)`
            2.  **FOR "AVERAGE RAINFALL" (by state):**
                - You MUST use `GROUP BY "SUBDIVISION"`.
                - **Example:** `SELECT "SUBDIVISION", AVG(ANNUAL) FROM rainfall ... GROUP BY "SUBDIVISION"`
            3.  **FOR "TOP N CROPS" (by state):**
                - You MUST `SELECT crop, SUM(production_tonnes)`.
                - You MUST `GROUP BY crop`.
                - You MUST `ORDER BY SUM(production_tonnes) DESC`.
                - **Example:** `SELECT crop, SUM(production_tonnes) AS total_production FROM crop_production ... GROUP BY crop ORDER BY total_production DESC LIMIT 3`
            4. **FOR "CROP NAME" ERRORS:**
                 - If you query for a crop by name (e.g., `WHERE crop = 'Rice'`) and get an empty result,
                   your *first* corrective action MUST be to query the `crop` column
                   (e.g., `SELECT DISTINCT crop FROM crop_production WHERE crop LIKE '%Rice%'`)
                   to find the correct database spelling (e.g., 'Rice (Paddy)') and then
                   re-run your original query with the correct name.
            5. If I've tried to get all the data, and some of it is missing or sparse, stop looping anyway and just report the data I was able to find.

            **RULES:**
            - Only output the raw SQL query. Do NOT add markdown, backticks, or any explanation.
            """
        ),
        ("human", "{question}"),
    ]
)

sql_chain = (
    {"question": RunnablePassthrough()}
    | RunnablePassthrough.assign(schema=lambda x: DB_SCHEMA)
    | sql_prompt
    | llm
    | StrOutputParser()
    | run_sql_query
    | parse_sql_result
)

AGENT_PREFIX = """
You are a data analyst agent. You are interacting with a SQLite database.
You are a fallback 'expert' agent. You are only called when the primary, fast tool fails.
This means the query is likely very complex. Take your time, list the tables,
check the schema, and build your query step-by-step.

**CRITICAL SCHEMA NOTES:**
- 'rainfall' table: 'SUBDIVISION' is state, 'ANNUAL' is total rainfall.
- 'crop_production' table: 'state_name' is state, 'crop' is crop name, 'production_tonnes' is volume.

**CRITICAL LOGIC NOTES:**

1.  **FOR "LAST N YEARS" (Crops):**
    - `... WHERE crop_year IN (SELECT DISTINCT crop_year FROM crop_production ORDER BY crop_year DESC LIMIT N)`

2.  **FOR "AVERAGE RAINFALL" (by state):**
    - You MUST use `GROUP BY "SUBDIVISION"`.
    - **Example:** `SELECT "SUBDIVISION", AVG(ANNUAL) FROM rainfall ... GROUP BY "SUBDIVISION"`
3.  **FOR "TOP N CROPS" (by state):**
    - You MUST `SELECT crop, SUM(production_tonnes)`.
    - You MUST `GROUP BY crop`.
    - You MUST `ORDER BY SUM(production_tonnes) DESC`.
    - **Example:** `SELECT crop, SUM(production_tonnes) AS total_production FROM crop_production ... GROUP BY crop ORDER BY total_production DESC LIMIT 3`

4.  **FOR "CROP NAME" ERRORS:**
    - If you query for a crop by name (e.g., `WHERE crop = 'Rice'`) and get an empty result,
      your *first* corrective action MUST be to query the `crop` column
      (e.g., `SELECT DISTINCT crop FROM crop_production WHERE crop LIKE '%Rice%'`)
      to find the correct database spelling (e.g., 'Rice (Paddy)') and then
      re-run your original query with the correct name.
5. If I've tried to get all the data, and some of it is missing or sparse, stop looping anyway and just report the data I was able to find.
6. When you execute a SQL query, you must only report the exact data returned from the query. If the query returns an empty result, you must state 'No data found' or return an empty string. You are strictly forbidden from inventing, inferring, or hallucinating an answer that is not explicitly present in the database response.
7. Pay close attention to the provided schema and example rows. The user may provide a district name, state name, or crop name. You must ensure you query the correct column for the given value (e.g., query district_name for districts, state_name for states).
**RULES:**
- You have access to tools for interacting with the database.
- You must follow the "Thought, Action, Observation" format.
- After you have the final answer, respond *only* with "Final Answer: [your_answer]".
- If the user explicitly mentions a district in the query, use the `district_name` column in the `crop_production` table.
"""

sql_agent_executor = create_sql_agent(
    llm=llm,
    db=db,
    agent_type="openai-tools",
    verbose=True,
    prefix=AGENT_PREFIX,
    handle_parsing_errors=True
)

def master_sql_executor(question: str) -> str:
    """
    Executes a SQL query. Tries the fast, simple chain first.
    If that fails (raises an Exception OR returns an empty/invalid result),
    it falls back to the slower, smarter ReAct agent.
    """
    try:
        
        print("--- ATTEMPTING FAST SQL CHAIN (1 API Call) ---")
        result = sql_chain.invoke(question)

        
        if result == "No data found." or result == "" or result == "[]":
            print(f"--- FAST SQL SUCCEEDED but returned empty. Triggering fallback for semantic check. ---")
            # We raise an exception *intentionally* to enter the fallback block
            raise ValueError("Fast chain returned an empty result, escalating to smart agent.")
        # --- END FIX ---

        print("--- FAST SQL SUCCEEDED ---")
        return result

    except (ValueError, sqlite3.OperationalError, Exception) as e:
        # --- 2. FALLBACK TO SMART AGENT ---
        print(f"--- FAST SQL FAILED (Reason: {e}) ---")
        print("--- FALLING BACK TO SMART SQL AGENT (10+ API Calls) ---")

        
        agent_result = sql_agent_executor.invoke({"input": question})

        
        answer = agent_result['output']
        print("--- SMART SQL AGENT FINISHED ---")
        return answer
        return agent_result['output']


def master_sql_executor(question: str) -> str:
    print(f"---SQL MASTER EXECUTOR: Running fast chain for: {question}---")
    try:
        result = sql_chain.invoke(question)
        
        # This fix catches None, [], "", and "No data found."
        if not result or result == "No data found.":
            raise ValueError("Fast chain returned an empty/None result, escalating to smart agent.")
        
        print("---SQL MASTER EXECUTOR: Fast chain success.---")
        return result
    except (ValueError, sqlite3.OperationalError, Exception) as e:
        # This code will NOW run correctly
        print(f"---SQL MASTER EXECUTOR: Fast chain failed ({e}). Escalating to smart agent.---")
        agent_result = sql_agent_executor.invoke({"input": question})
        return agent_result['output']


sql_agent_tool = Tool(
    name="SQL_Agent",
    func=master_sql_executor,
    description="A tool that can answer quantitative questions by querying the samarth.db database. Use this for all questions about crops, rainfall, production, states, districts, etc."
)

# --- RAG AGENT TOOL REMOVED ---

tools_dict = {
    "SQL_Agent": sql_agent_tool,
}

# --- CROP/DISTRICT/STATE CORRECTOR FUNCTIONS REMOVED (we will rely on the agent's fuzzy search) ---

QUERY_REWRITER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a query-rewriting assistant. Your *only* job is to
            take a 'Follow-up question' and 'Relevant Chat History' and
            rephrase the follow-up question into a clear, standalone question.

            **CRITICAL RULES:**
            1.  **DO NOT** answer the question yourself.
            2.  **DO NOT** add any new information.
            3.  If the 'Follow-up question' is already standalone, just return it exactly as-is.
            4.  Your output MUST be *only* the rewritten, standalone question.

            **Relevant Chat History:**
            {chat_history}
            """,
        ),
        ("human", "Follow-up question: {question}"),
    ]
)

query_rewriter_chain = QUERY_REWRITER_PROMPT | llm | StrOutputParser()

# Simple, low-memory rewriter (no embeddings)
def rewrite_query(state: AgentState):
    print("---NODE: rewrite_query (SLIM)---")
    user_query = state["user_query"]
    chat_history = state["chat_history"]
    
    if not chat_history:
        print("---REWRITER: No history, returning original query.---")
        return {"rewritten_query": user_query}

    formatted_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history[-5:]]) # Get last 5 messages

    try:
        rewritten_query = query_rewriter_chain.invoke({"chat_history": formatted_history, "question": user_query})
        print(f"---REWRITER: Original: '{user_query}' -> Rewritten: '{rewritten_query}'---")
    except Exception as e:
        print(f"---REWRITER: Failed to rewrite, using original query. Error: {e}")
        rewritten_query = user_query
        
    return {"rewritten_query": rewritten_query}


# "One-Shot" Orchestrator (prevents loops)
orchestrator_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert orchestrator for a data analysis chatbot.
            Your job is to break down a complex user query into a step-by-step plan.
            You will be given the user's query, chat history, and a list of data that has already been collected.

            You have two tools available:
            1.  **SQL_Agent**: Use this for **quantitative questions** about the data. (e.g., "What is the average rainfall...", "List the top 3 crops...", "What are the last 5 available years?")
            2.  **RAG_Agent**: Use this *only* for **metadata questions**. (e.g., "What is the source URL...", "Describe the rainfall dataset.")

            **CRITICAL RULE 1: CHOOSE THE RIGHT TOOL.**
            - Any question that asks for *values, numbers, lists, or dates* from the database (like "What is..." or "How many...") MUST go to the `SQL_Agent`.
            - Any question about *dataset sources or descriptions* MUST go to the `RAG_Agent`.
            - **DO NOT** ask the `RAG_Agent` a quantitative question. It will fail, and you must not try again.

            **CRITICAL RULE 2: CHECK COLLECTED DATA.**
            - Before you call a tool, review the `collected_data` list. If the information you
              need (based on the 'question_asked' and 'answer_received') is *already* in the list,
              **do not call the tool again**. Move on to the next step.

            **CRITICAL RULE 3: HANDLE IMPOSSIBLE TASKS.**
            - If a tool tells you data is unavailable (e.g., "data for 2024 does not exist"),
              **stop trying to get it**. Acknowledge this limitation and move to `SYNTHESIZE`.

            **CRITICAL RULE 4: ONE THING AT A TIME.**
            - Ask for one piece of information at a time (e.g., rainfall for one state).

            **CRITICAL RULE 5: RAG TOOL USAGE.**
              - The RAG_Agent only knows about *general* dataset metadata.
              - **DO NOT** ask it for specific regions (like 'Bihar' or 'Andaman').
              - **ALWAYS** ask general questions like: "What is the source for all rainfall data?"

            Respond with a JSON object with your decision:
            {{
                "decision": "CALL_TOOL" or "SYNTHESIZE",
                "tool_name": "SQL_Agent" or "RAG_Agent" (if calling tool),
                "tool_input": "The specific, single question." (if calling tool),
                "reasoning": "Your brief thought process for this *single* step."
            }}

            **Example of correct tool selection:**
            User Query: "What was the rainfall in 2009 and what is the source?"
            Collected Data: []
            Your JSON Output:
            {{
                "decision": "CALL_TOOL",
                "tool_name": "SQL_Agent",
                "tool_input": "What was the annual rainfall in 2009?",
                "reasoning": "First, I must use the SQL_Agent to get the quantitative rainfall value."
            }}

            **After that tool call:**
            Collected Data: [{{"tool_called": "SQL_Agent", "question_asked": "What was the annual rainfall in 2009?", "answer_received": "2538.6"}}]
            Your JSON Output:
            {{
                "decision": "CALL_TOOL",
                "tool_name": "RAG_Agent",
                "tool_input": "What is the source for the rainfall data?",
                "reasoning": "I have the rainfall value (2538.6). Now I will use the RAG_Agent to get the metadata (source)."
            }}
            """
        ),
        ("human", "User Query: {user_query}\nChat History: {chat_history}\n\nCollected Data: {collected_data}"),
    ]
)

orchestrator_chain = orchestrator_prompt | llm | JsonOutputParser()

def plan_workflow(state: AgentState):
    print("---NODE: plan_workflow---")
    
    # NEW RULE: If we have data, just synthesize. This stops loops.
    if state["collected_data"]:
        print("---PLANNER: Data found. Escalating to SYNTHESIZE.---")
        return {"planner_decision": {"decision": "SYNTHESIZE", "tool_name": None, "tool_input": None, "reasoning": "Data has been collected, synthesizing answer."}}
        
    prompt_input = {
        "user_query": state['rewritten_query'],
        "chat_history": state['chat_history'],
        "collected_data": state['collected_data']
    }
    try:
        response = orchestrator_chain.invoke(prompt_input)
    except Exception as e:
        print(f"---PLANNER: Error, escalating to SYNTHESIZE. {e}---")
        response = {"decision": "SYNTHESIZE", "tool_name": None, "tool_input": None, "reasoning": "Faced an error during planning, attempting to synthesize a response."}
    print(f"---PLANNER: Decision: {response}---")
    return {"planner_decision": response}

# This node is now just a pass-through, but we keep it for graph structure
def correct_sql_input(state: AgentState):
    print("---NODE: correct_sql_input (Passing through)---")
    return # No change, proceed to execute_tool

def execute_tool(state: AgentState):
    print("---NODE: execute_tool---")
    decision = state["planner_decision"]
    tool_name = decision["tool_name"]
    tool_input = decision["tool_input"]
    
    print(f"---TOOL EXECUTOR: Calling {tool_name} with input: {tool_input}---")
    tool_to_run = tools_dict[tool_name]
    result = tool_to_run.invoke(tool_input)
    print(f"---TOOL EXECUTOR: Result: {result}---")

    structured_data_point = {"tool_called": tool_name, "question_asked": tool_input, "answer_received": result}
    new_data_list = state["collected_data"] + [structured_data_point]
    return {"collected_data": new_data_list}

def route_from_planner(state: AgentState):
    print("---NODE: route_from_planner---")
    decision = state["planner_decision"]["decision"]
    
    if decision == "SYNTHESIZE":
        print("---ROUTER: -> synthesize_answer---")
        return "synthesize_answer"
    
    if decision == "CALL_TOOL":
        tool_name = state["planner_decision"]["tool_name"]
        if tool_name == "SQL_Agent":
            print("---ROUTER: -> correct_sql_input---")
            return "correct_sql_input"
            
    print("---ROUTER: Defaulting -> synthesize_answer---")
    return "synthesize_answer"


# --- THIS IS THE FINAL, "SMART" SYNTHESIZER ---
synthesizer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful assistant. Your job is to give a final, direct answer to the user.

            **Rules:**
            1.  Focus *only* on the user's most recent query: **{user_query}**
            2.  Use the "All Collected Data" (from the SQL_Agent) to find the answer. Do not mention the data collection process.
            3.  **DO NOT HALLUCINATE.** If the data contradicts the user's premise (e.g., they want to "support" a policy but the data is negative), you MUST present the data objectively as-is. It is your job to be truthful to the data.
            4.  Be brief, natural, and conversational.
            5.  Acknowledge failures if data was unavailable (e.g., "No data was found for 2024.").

            **CRITICAL RULE: CITE SOURCES**
            After giving your answer, you MUST check the user's query ({user_query}) and the collected data for keywords.
            - If the query or data mentions "crop" or "production", add this citation:
              *Source (Crop Data): https://www.data.gov.in/catalog/district-wise-season-wise-crop-production-statistics-0*
            - If the query or data mentions "rainfall" or "climate", add this citation:
              *Source (Rainfall Data): https://www.data.gov.in/resource/sub-divisional-monthly-rainfall-1901-2017#api*
            - If both topics are mentioned, cite both.
            - If the query is a simple greeting (like "hello"), do not add sources.

            **Chat History (for context only):**
            {chat_history}

            **All Collected Data (to answer the query):**
            {collected_data}

            Generate the final, direct answer, and add the correct source citations below it.
            """
        ),
    ]
)
synthesizer_chain = synthesizer_prompt | llm | StrOutputParser()
# --- END OF "SMART" SYNTHESIZER ---


def synthesize_answer(state: AgentState):
    print("---NODE: synthesize_answer---")
    formatted_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in state["chat_history"]])
    
    # We pass the *original* user_query to the synthesizer so it can check for keywords
    prompt_input = {"user_query": state['user_query'], "chat_history": formatted_history, "collected_data": state['collected_data']}
    
    try:
        final_response = synthesizer_chain.invoke(prompt_input)
    except Exception as e:
        print(f"---SYNTHESIZER: Error: {e}---")
        final_response = "Sorry, I ran into an error while generating the final answer."
    
    print(f"---SYNTHESIZER: Final Response: {final_response}---")
    return {"final_response": final_response}

# --- Build and Compile the Graph ---
print("Compiling LangGraph workflow (SLIM-BUT-SMART Mode)...")
workflow = StateGraph(AgentState)
workflow.add_node("rewrite_query", rewrite_query)
workflow.add_node("plan_workflow", plan_workflow)
workflow.add_node("correct_sql_input", correct_sql_input)
workflow.add_node("execute_tool", execute_tool)
workflow.add_node("synthesize_answer", synthesize_answer)

workflow.set_entry_point("rewrite_query")
workflow.add_edge("rewrite_query", "plan_workflow")
workflow.add_conditional_edges("plan_workflow", route_from_planner, {
    "synthesize_answer": "synthesize_answer",
    "correct_sql_input": "correct_sql_input",
})
workflow.add_edge("correct_sql_input", "execute_tool")
workflow.add_edge("execute_tool", "plan_workflow") # Loop back to planner
workflow.add_edge("synthesize_answer", END)


# This is the final, compiled app that main.py will import
agent_app = workflow.compile()
print("LangGraph workflow compiled successfully. Ready to serve (SLIM-BUT-SMART).")