import os
from typing import TypedDict, List

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

from tools import web_search, fetch_page, summarise


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# GEMINI MODEL
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)


# ============================================================
# HARD STEP LIMIT
# ============================================================

MAX_STEPS = 6


# ============================================================
# AGENT STATE
# ============================================================

class AgentState(TypedDict):

    question: str

    selected_tool: str

    search_results: str

    fetched_sources: List[str]

    summaries: List[str]

    final_answer: str

    steps: int


# ============================================================
# NODE 1: AGENT DECIDES WHICH TOOL TO USE
# ============================================================

def decide_tool_node(state: AgentState):

    question = state["question"]

    print("\n🧠 Agent deciding which tool to use...")

    prompt = f"""
You are a research agent.

User question:
{question}

Available tools:

1. SEARCH
Use SEARCH when you need to find information or sources
on the web.

2. FETCH
Use FETCH when you already have a webpage URL and need
the actual webpage content.

Choose exactly ONE tool.

Return ONLY one word:
SEARCH
or
FETCH
"""

    try:

        response = llm.invoke(prompt)

        decision = response.content.strip().upper()

        if "FETCH" in decision:
            selected_tool = "fetch"

        else:
            selected_tool = "search"

    except Exception:

        # Safe fallback
        selected_tool = "search"

    print(f"🔧 Selected tool: {selected_tool}")

    return {
        "selected_tool": selected_tool,
        "steps": state["steps"] + 1
    }


# ============================================================
# NODE 2: WEB SEARCH
# ============================================================

def search_node(state: AgentState):

    print("\n🔎 Running web search...")

    question = state["question"]

    result = web_search(question)

    if not result or result.startswith("Search tool failed"):
        print("⚠️ Search failed.")

        return {
            "search_results": "",
            "steps": state["steps"] + 1
        }

    return {
        "search_results": result,
        "steps": state["steps"] + 1
    }


# ============================================================
# NODE 3: FETCH WEBPAGE
# ============================================================

def fetch_node(state: AgentState):

    print("\n🌐 Running fetch tool...")

    search_results = state["search_results"]

    if not search_results:

        print("⚠️ No search results available.")

        return {
            "fetched_sources": [],
            "steps": state["steps"] + 1
        }

    # Find first URL from search results
    lines = search_results.splitlines()

    url = None

    for line in lines:

        if line.startswith("URL:"):

            url = line.replace(
                "URL:",
                ""
            ).strip()

            break

    if not url:

        print("⚠️ No URL found.")

        return {
            "fetched_sources": [],
            "steps": state["steps"] + 1
        }

    print(f"📄 Fetching: {url}")

    page = fetch_page(url)

    if page.startswith("Fetch tool failed"):

        print("⚠️ Fetch failed.")

        return {
            "fetched_sources": [],
            "steps": state["steps"] + 1
        }

    if not page.strip():

        print("⚠️ Empty webpage.")

        return {
            "fetched_sources": [],
            "steps": state["steps"] + 1
        }

    return {
        "fetched_sources": [page],
        "steps": state["steps"] + 1
    }


# ============================================================
# NODE 4: SUMMARISE
# ============================================================

def summarise_node(state: AgentState):

    print("\n📝 Summarising fetched source...")

    sources = state["fetched_sources"]

    if not sources:

        return {
            "summaries": [],
            "steps": state["steps"] + 1
        }

    summaries = []

    for source in sources:

        summary = summarise(source)

        if summary:
            summaries.append(summary)

    return {
        "summaries": summaries,
        "steps": state["steps"] + 1
    }


# ============================================================
# NODE 5: FINAL ANSWER
# ============================================================

def final_answer_node(state: AgentState):

    print("\n🤖 Generating final answer...")

    question = state["question"]

    summaries = state["summaries"]

    if not summaries:

        return {
            "final_answer": (
                "Sorry, I could not find reliable "
                "source information for this question."
            )
        }

    evidence = "\n\n".join(summaries)

    prompt = f"""
You are a research assistant.

Answer the following research question using ONLY
the provided source evidence.

Research Question:
{question}

Source Evidence:
{evidence}

Rules:

1. Do not invent facts.
2. Do not use information outside the evidence.
3. Give a clear and concise answer.
4. If evidence is insufficient, say so.
5. Mention that the answer is based on the fetched source.
"""

    try:

        response = llm.invoke(prompt)

        return {
            "final_answer": response.content
        }

    except Exception as e:

        return {
            "final_answer": (
                f"Unable to generate final answer: {e}"
            )
        }


# ============================================================
# ROUTING: AGENT TOOL DECISION
# ============================================================

def route_tool(state: AgentState):

    if state["steps"] >= MAX_STEPS:

        print("\n⛔ Maximum step limit reached.")

        return "final"

    selected_tool = state["selected_tool"]

    if selected_tool == "fetch":

        return "fetch"

    return "search"


# ============================================================
# BUILD LANGGRAPH
# ============================================================

graph = StateGraph(AgentState)


# Add nodes
graph.add_node(
    "decide",
    decide_tool_node
)

graph.add_node(
    "search",
    search_node
)

graph.add_node(
    "fetch",
    fetch_node
)

graph.add_node(
    "summarise",
    summarise_node
)

graph.add_node(
    "final",
    final_answer_node
)


# Entry point
graph.set_entry_point("decide")


# Agent decides between tools
graph.add_conditional_edges(
    "decide",
    route_tool,
    {
        "search": "search",
        "fetch": "fetch",
        "final": "final"
    }
)


# Search → Fetch
graph.add_edge(
    "search",
    "fetch"
)


# Fetch → Summarise
graph.add_edge(
    "fetch",
    "summarise"
)


# Summarise → Final
graph.add_edge(
    "summarise",
    "final"
)


# Final → End
graph.add_edge(
    "final",
    END
)


# Compile graph
research_agent = graph.compile()


# ============================================================
# RUN RESEARCH AGENT
# ============================================================

def run_research_agent(question: str):

    initial_state = {

        "question": question,

        "selected_tool": "",

        "search_results": "",

        "fetched_sources": [],

        "summaries": [],

        "final_answer": "",

        "steps": 0
    }

    result = research_agent.invoke(
        initial_state
    )

    return result


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print("\n========================================")
    print("      AGENTICX RESEARCH AGENT")
    print("========================================")

    question = input(
        "\nEnter your research question: "
    )

    if not question.strip():

        print(
            "\n❌ Please enter a research question."
        )

    else:

        result = run_research_agent(
            question
        )

        print("\n========================================")
        print("             FINAL ANSWER")
        print("========================================\n")

        print(
            result["final_answer"]
        )

        print("\n========================================")
        print(
            f"Steps used: {result['steps']}"
        )
        print(
            f"Tool selected: {result['selected_tool']}"
        )
        print("========================================")