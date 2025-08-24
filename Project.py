import os
from dotenv import load_dotenv
from crewai import Agent, Crew, Process, Task, LLM
from crewai.tools import tool
from duckduckgo_search import DDGS
from phi.model.groq import Groq

# Load API keys from .env file
load_dotenv()
GROQ_API_KEY = os.getenv("Groq_api_key")

# ----------------- LLMs -----------------
llm_groq_creative = LLM(
    model="groq/llama-3.3-70b-versatile",
    temperature=1.0,
    api_key=GROQ_API_KEY,
)

llm_openai_precise = LLM(
    model="gpt-4o-mini",
    temperature=0.9,
    api_key=GROQ_API_KEY,
)

# ----------------- Tools -----------------
@tool("query_duckduckgo")
def query_duckduckgo(query: str, max_results: int = 5) -> str:
    """
    Queries the internet using DuckDuckGo search engine and returns results.

    Parameters:
    - query (str): The search query.
    - max_results (int, optional): Number of results to fetch. Default = 5.

    Returns:
    - str: Combined text response of search results.
    """
    results = DDGS().text(query, max_results=max_results, region="us-en")
    output = []
    for r in results:
        output.append(f"Title: {r['title']}\nLink: {r['href']}\nSnippet: {r['body']}\n")
    return "\n".join(output)


@tool("save_markdown")
def save_markdown(content: str, filename: str = "renewables_report.md") -> str:
    """
    Saves given content into a markdown file.

    Parameters:
    - content (str): The markdown content.
    - filename (str, optional): File name to save. Default = renewables_report.md

    Returns:
    - str: Confirmation message with file path.
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    return f"✅ Report saved as {filename}"

# ----------------- Agents -----------------
research_agent = Agent(
    role="Research Analyst",
    goal="Find latest information about renewable energy.",
    backstory="Expert at searching and summarizing technical information.",
    llm=llm_groq_creative,
    tools=[query_duckduckgo],
    verbose=True
)

writer_agent = Agent(
    role="Content Writer",
    goal="Write a detailed report on renewable energy.",
    backstory="Skilled at turning raw research into engaging reports.",
    llm=llm_groq_creative,
    tools=[save_markdown],
    verbose=True
)

# ----------------- Tasks -----------------
search_task = Task(
    description="Search the internet for the latest information on renewable energy in 2025.",
    expected_output="A summary of top renewable energy trends with references.",
    agent=research_agent,
    output_file="search_results.md"
)

report_task = Task(
    description="Write a structured research report based on the findings.",
    expected_output="A markdown research report about renewable energy adoption.",
    agent=writer_agent,
    context=[search_task],
    output_file="renewables_report.md"
)

save_task = Task(
    description="Save the generated report into a markdown file.",
    expected_output="File saved confirmation.",
    agent=writer_agent,
    context=[report_task]
)

# ----------------- Crew -----------------
crew = Crew(
    agents=[research_agent, writer_agent],
    tasks=[search_task, report_task, save_task],
    process=Process.sequential,
    verbose=True
)

if __name__ == "__main__":
    result = crew.kickoff()
    print(result)
