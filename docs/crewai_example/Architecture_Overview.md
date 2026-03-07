Title: Architecture Overview - Company Research Agent

URL Source: https://company-research-agent.mintlify.app/architecture/overview

Markdown Content:
The Company Research Agent is built using CrewAI, a powerful framework for orchestrating multiple AI agents to work together on complex tasks. The architecture combines specialized agents, custom tools, and the Apify platform to deliver comprehensive company research.

High-Level Architecture
-----------------------

Core Components
---------------

### 1. CrewAI Crew

The `CompanyResearchCrew` class orchestrates three specialized agents:

```
@CrewBase
class CompanyResearchCrew:
    def crew(self) -> Crew:
        return Crew(
            agents=[
                self.researcher(),
                self.data_analyst(),
            ],
            tasks=[
                self.research_company(),
                self.analyze_data(),
            ],
            process=Process.sequential
        )
```

### 2. Specialized Agents

Each agent has a specific role in the research process:

*   **Researcher**: Gathers raw company information
*   **Data Analyst**: Processes and analyzes the collected data
*   **Content Compiler**: Formats the findings into structured reports

### 3. Custom Tools

The system includes specialized tools for different data sources:

```
tools=[
    CompanyNewsSearchTool(actor=self.actor),
    ProfessionalProfilesTool(actor=self.actor),
    LinkedInScraperTool(actor=self.actor),
    CrunchbaseScraperTool(actor=self.actor),
    PitchBookScraperTool(actor=self.actor),
    GoogleSearchTool(actor=self.actor)
]
```

### 4. Task Flow

The research process follows a sequential workflow:

1.   Company research using domain name
2.   Data analysis and insight extraction
3.   Report compilation and formatting

Integration Points
------------------

### Apify Platform

*   **Actors**: Uses specialized Apify actors for web scraping
*   **Dataset**: Stores and manages research results
*   **Key-Value Store**: Handles intermediate data storage

### LLM Integration

```
self.llm = LLM(
    model="gemini/gemini-2.0-flash-lite",
    temperature=0.7,
    api_key=os.getenv("GOOGLE_API_KEY")
)
```

Data Flow
---------

1.   **Input**: Company domain name
2.   **Research Phase**: Multiple tools gather data in parallel
3.   **Analysis Phase**: Data is processed and analyzed
4.   **Output**: Structured report in JSON format

Error Handling
--------------

The system includes robust error handling:

*   Tool-level error handling for each data source
*   Agent-level task retry mechanisms
*   Crew-level process management

Performance Optimization
------------------------

*   Concurrent data collection where possible
*   Rate limiting controls
*   Memory management for large datasets