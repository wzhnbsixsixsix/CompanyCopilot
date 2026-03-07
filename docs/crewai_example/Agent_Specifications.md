Title: Agent Specifications - Company Research Agent

URL Source: https://company-research-agent.mintlify.app/architecture/agents

Markdown Content:
The Company Research Agent uses three specialized AI agents, each with a specific role in the research process. This document details each agent’s configuration and responsibilities.

Researcher Agent
----------------

The primary data gathering agent.

```
@agent
def researcher(self) -> Agent:
    """
    @agent Company Research Specialist
    @description Expert in gathering comprehensive company information
    @returns Agent - Configured researcher agent
    """
    return Agent(
        role="Company Research Specialist",
        goal="Gather comprehensive information about companies from various sources",
        backstory="""You are an expert business researcher with years of experience
        in gathering and analyzing company information. You excel at finding accurate
        and relevant details about organizations, their products, and market presence.""",
        tools=[
            CompanyNewsSearchTool(actor=self.actor),
            ProfessionalProfilesTool(actor=self.actor),
            LinkedInScraperTool(actor=self.actor),
            CrunchbaseScraperTool(actor=self.actor),
            PitchBookScraperTool(actor=self.actor),
            GoogleSearchTool(actor=self.actor)
        ],
        llm=self.llm
    )
```

### Researcher Capabilities

Data Analyst Agent
------------------

Processes and analyzes gathered information.

```
@agent
def data_analyst(self) -> Agent:
    """
    @agent Business Data Analyst
    @description Expert in analyzing company data and extracting insights
    @returns Agent - Configured analyst agent
    """
    return Agent(
        role="Business Data Analyst",
        goal="Analyze company data and extract meaningful insights",
        backstory="""You are a skilled data analyst specializing in business metrics
        and market analysis. You have a strong background in interpreting company 
        performance data and identifying market trends.""",
        llm=self.llm
    )
```

### Analysis Capabilities

*   Market trend identification
*   Financial metric analysis
*   Competitive landscape assessment
*   Growth pattern recognition
*   Risk factor identification

Content Compiler Agent
----------------------

Formats and structures the research findings.

```
@agent
def content_compiler(self) -> Agent:
    """
    @agent Business Report Writer
    @description Expert in creating comprehensive business reports
    @returns Agent - Configured compiler agent
    """
    return Agent(
        role="Business Report Writer",
        goal="Compile research findings into comprehensive, well-structured reports",
        backstory="""You are an experienced business writer who excels at organizing
        complex information into clear, actionable reports. You have a keen eye for
        important details and can present information in a professional format.""",
        llm=self.llm
    )
```

### Compilation Capabilities

*   Report structuring
*   Key insight highlighting
*   Data visualization recommendations
*   Executive summary creation
*   Action item identification

Agent Interaction Flow
----------------------

Configuration Parameters
------------------------

### Common Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| role | string | Agent’s specific role |
| goal | string | Primary objective |
| backstory | string | Context and expertise |
| llm | LLM | Language model instance |

### Researcher-Specific Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| tools | List[BaseTool] | Available research tools |
| verbose | boolean | Output verbosity |

Best Practices
--------------

1.   **Agent Independence**
    *   Each agent operates independently
    *   Clear separation of responsibilities
    *   Minimal cross-agent dependencies

2.   **Error Handling**
    *   Agents handle task-specific errors
    *   Graceful failure recovery
    *   Clear error reporting

3.   **Performance Optimization**
    *   Efficient tool usage
    *   Resource management
    *   Output filtering