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
    source_urls: List[str]
    source_titles: List[str]
    summaries: List[str]
    final_answer: str
    steps: int


# ============================================================
# NODE 1: DECIDE TOOL
# ============================================================

def decide_tool_node(state: AgentState):

    print("\n🧠 Agent deciding which tool to use...")

    question = state["question"]

    prompt = f"""
You are a research agent.

User question:
{question}

Available tools:

SEARCH:
Use this when you need to find information on the web.

FETCH:
Use this when a webpage URL is already available and
you need the webpage content.

Choose exactly one.

Return only:
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
            "source_urls": [],
            "source_titles": [],
            "steps": state["steps"] + 1
        }

    # Extract source URLs and titles
    lines = result.splitlines()

    urls = []
    titles = []

    for line in lines:

        if line.startswith("Title:"):
            title = line.replace(
                "Title:",
                ""
            ).strip()

            titles.append(title)

        elif line.startswith("URL:"):
            url = line.replace(
                "URL:",
                ""
            ).strip()

            if url:
                urls.append(url)

    print(f"📚 Sources found: {len(urls)}")

    return {
        "search_results": result,
        "source_urls": urls,
        "source_titles": titles,
        "steps": state["steps"] + 1
    }


# ============================================================
# NODE 3: FETCH SOURCE
# ============================================================

def fetch_node(state: AgentState):

    print("\n🌐 Running fetch tool...")

    urls = state["source_urls"]

    if not urls:

        print("⚠️ No source URL available.")

        return {
            "fetched_sources": [],
            "steps": state["steps"] + 1
        }

    # Fetch first source
    url = urls[0]

    print(f"📄 Fetching source:")
    print(url)

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
# NODE 5: FINAL ANSWER WITH SOURCES
# ============================================================

def final_answer_node(state: AgentState):

    print("\n🤖 Generating final answer...")

    question = state["question"]

    summaries = state["summaries"]

    urls = state["source_urls"]

    titles = state["source_titles"]

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

Answer the research question using ONLY the provided
source evidence.

Research Question:
{question}

Source Evidence:
{evidence}

Rules:
1. Do not invent facts.
2. Do not use outside knowledge.
3. Every factual claim must be supported by the evidence.
4. If the evidence is insufficient, clearly say so.
5. Give a concise and useful answer.
"""

    try:

        response = llm.invoke(prompt)

        answer = response.content

        # Add source section
        source_section = "\n\n### Sources\n"

        for i, url in enumerate(urls):

            if i < len(titles):

                title = titles[i]

            else:

                title = "Source"

            source_section += (
                f"\n[{i + 1}] {title}\n"
                f"{url}\n"
            )

        final_output = (
            answer +
            source_section
        )

        return {
            "final_answer": final_output
        }

    except Exception as e:

        return {
            "final_answer": (
                f"Unable to generate final answer: {e}"
            )
        }


# ============================================================
# ROUTING
# ============================================================

def route_tool(state: AgentState):

    if state["steps"] >= MAX_STEPS:

        print("\n⛔ Maximum step limit reached.")

        return "final"

    if state["selected_tool"] == "fetch":

        return "fetch"

    return "search"


# ============================================================
# BUILD LANGGRAPH
# ============================================================

graph = StateGraph(AgentState)


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


# Decide → Search OR Fetch
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


# Compile
research_agent = graph.compile()


# ============================================================
# RUN AGENT
# ============================================================

def run_research_agent(question: str):

    initial_state = {

        "question": question,

        "selected_tool": "",

        "search_results": "",

        "fetched_sources": [],

        "source_urls": [],

        "source_titles": [],

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
        print("========================================")