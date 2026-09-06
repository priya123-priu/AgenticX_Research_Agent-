import os
import requests

from dotenv import load_dotenv
from tavily import TavilyClient
from bs4 import BeautifulSoup


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# CREATE TAVILY CLIENT
# ============================================================

tavily = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)


# ============================================================
# TOOL 1: WEB SEARCH
# ============================================================

def web_search(query: str) -> str:
    """
    Search the web using Tavily
    and return source information.
    """

    try:
        response = tavily.search(
            query=query,
            search_depth="advanced",
            max_results=5
        )

        results = response.get("results", [])

        if not results:
            return "No search results found."

        output = []

        for result in results:

            title = result.get(
                "title",
                "Unknown"
            )

            url = result.get(
                "url",
                ""
            )

            content = result.get(
                "content",
                ""
            )

            output.append(
                f"Title: {title}\n"
                f"URL: {url}\n"
                f"Content: {content}"
            )

        return "\n\n---\n\n".join(output)

    except Exception as e:

        return f"Search tool failed: {e}"


# ============================================================
# TOOL 2: FETCH WEBPAGE
# ============================================================

def fetch_page(url: str) -> str:
    """
    Fetch readable text from a webpage.
    """

    try:

        response = requests.get(
            url,
            timeout=10,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        # Remove unnecessary HTML elements
        for tag in soup(
            ["script", "style", "nav", "footer"]
        ):
            tag.decompose()

        # Extract readable text
        text = soup.get_text(
            separator=" ",
            strip=True
        )

        if not text:

            return "No readable content found on this page."

        # Limit text size
        return text[:12000]

    except requests.RequestException as e:

        return f"Fetch tool failed: {e}"

    except Exception as e:

        return f"Unexpected fetch error: {e}"


# ============================================================
# TOOL 3: SUMMARISE
# ============================================================

def summarise(text: str) -> str:
    """
    Create a simple summary from fetched text.
    """

    if not text or text.strip() == "":
        return "Nothing to summarise."

    # Split text into sentences
    sentences = text.split(".")

    summary = []

    # Take first 5 meaningful sentences
    for sentence in sentences:

        sentence = sentence.strip()

        if sentence:

            summary.append(sentence)

        if len(summary) == 5:
            break

    if not summary:

        return "Unable to create summary."

    return ". ".join(summary) + "."


# ============================================================
# TEST THE TOOLS
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # TEST 1: WEB SEARCH
    # --------------------------------------------------------

    print("\n========================================")
    print("       WEB SEARCH TEST")
    print("========================================\n")

    search_result = web_search(
        "latest applications of artificial intelligence in healthcare"
    )

    print(search_result)


    # --------------------------------------------------------
    # TEST 2: FETCH PAGE
    # --------------------------------------------------------

    print("\n========================================")
    print("       FETCH PAGE TEST")
    print("========================================\n")

    test_url = "https://www.example.com"

    page_content = fetch_page(test_url)

    print(page_content)


    # --------------------------------------------------------
    # TEST 3: SUMMARISE
    # --------------------------------------------------------

    print("\n========================================")
    print("       SUMMARY TEST")
    print("========================================\n")

    sample_text = """
    Artificial intelligence is widely used in healthcare.
    AI can help doctors analyze medical images.
    Machine learning can support disease prediction.
    AI systems can improve hospital workflows.
    These technologies are becoming increasingly important.
    """

    summary = summarise(sample_text)

    print(summary)


    # --------------------------------------------------------
    # ALL TESTS COMPLETED
    # --------------------------------------------------------

    print("\n========================================")
    print("       ALL TOOL TESTS COMPLETED")
    print("========================================")

