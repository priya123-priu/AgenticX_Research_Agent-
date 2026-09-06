# AgenticX Tool-Using Research Agent

## 📌 Project Overview

This project is a Tool-Using Research Agent built using Python,
LangGraph, Gemini, and Tavily.

The agent accepts a research question, selects an appropriate
research tool, searches the web, fetches webpage content,
summarises the source, and generates a final answer based on
the collected evidence.

## 🚀 Features

- AI-powered research assistant
- Web search using Tavily
- Webpage fetching
- Source content summarisation
- Gemini-powered reasoning
- LangGraph state-based workflow
- Tool selection
- Hard step limit to prevent infinite loops
- Graceful handling of failed or empty tool results
- Evidence-based final answers

## 🛠️ Technologies Used

- Python
- LangGraph
- LangChain
- Google Gemini
- Tavily Search API
- Requests
- BeautifulSoup
- python-dotenv

## 📂 Project Structure

```text
AgenticX_Research_Agent/
│
├── agent.py
├── tools.py
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .env
└── venv/