Title: Custom Tools - Company Research Agent

URL Source: https://company-research-agent.mintlify.app/architecture/tools

Markdown Content:
The Company Research Agent implements several custom tools that extend CrewAI’s `BaseTool` class. These tools provide specialized functionality for gathering company information from various sources.

Tool Overview
-------------

Company News Search Tool
------------------------

```
class CompanyNewsSearchTool(BaseTool):
    """
    @class CompanyNewsSearchTool
    @description Searches for and retrieves recent news articles about a company
    @param actor - Apify Actor instance
    """
    name: str = "Company News Search"
    description: str = """
    Searches for and retrieves recent news articles about a company using its domain name.
    Returns articles with titles, URLs, descriptions, and publication dates.
    """
    actor: Actor = Field(description="Apify Actor instance")
```

### Usage Example

```
news_tool = CompanyNewsSearchTool(actor=actor)
news_articles = news_tool._run("example.com")
```

Professional Profiles Tool
--------------------------

```
class ProfessionalProfilesTool(BaseTool):
    """
    @class ProfessionalProfilesTool
    @description Finds company profiles on professional platforms
    @param actor - Apify Actor instance
    """
    name: str = "Professional Profiles Search"
    description: str = """
    Finds company profiles on LinkedIn, Crunchbase, and PitchBook using domain name.
    Returns profile URLs and descriptions for each platform.
    """
    actor: Actor = Field(description="Apify Actor instance")
```

### Usage Example

```
profiles_tool = ProfessionalProfilesTool(actor=actor)
profiles = profiles_tool._run("example.com")
```

LinkedIn Scraper Tool
---------------------

```
class LinkedInScraperTool(BaseTool):
    """
    @class LinkedInScraperTool
    @description Scrapes detailed company information from LinkedIn
    @param actor - Apify Actor instance
    """
    name: str = "LinkedIn Company Profile Scraper"
    description: str = """
    Scrapes detailed company information from LinkedIn company profiles.
    Requires a valid LinkedIn company profile URL.
    """
    actor: Actor = Field(description="Apify Actor instance")
```

### Usage Example

```
linkedin_tool = LinkedInScraperTool(actor=actor)
company_data = linkedin_tool._run("https://linkedin.com/company/example")
```

Crunchbase Scraper Tool
-----------------------

```
class CrunchbaseScraperTool(BaseTool):
    """
    @class CrunchbaseScraperTool
    @description Scrapes company information from Crunchbase
    @param actor - Apify Actor instance
    """
    name: str = "Crunchbase Organization Scraper"
    description: str = """
    Scrapes detailed company information from Crunchbase organization profiles.
    Requires a valid Crunchbase organization URL.
    """
    actor: Actor = Field(description="Apify Actor instance")
```

### Usage Example

```
crunchbase_tool = CrunchbaseScraperTool(actor=actor)
funding_data = crunchbase_tool._run("https://crunchbase.com/organization/example")
```

PitchBook Scraper Tool
----------------------

```
class PitchBookScraperTool(BaseTool):
    """
    @class PitchBookScraperTool
    @description Scrapes company information from PitchBook
    @param actor - Apify Actor instance
    """
    name: str = "PitchBook Company Profile Scraper"
    description: str = """
    Scrapes detailed company information from PitchBook company profiles.
    Requires a valid PitchBook company profile URL.
    """
    actor: Actor = Field(description="Apify Actor instance")
```

### Usage Example

```
pitchbook_tool = PitchBookScraperTool(actor=actor)
company_info = pitchbook_tool._run("https://pitchbook.com/profiles/company/example")
```

Google Search Tool
------------------

```
class GoogleSearchTool(BaseTool):
    """
    @class GoogleSearchTool
    @description Performs Google searches for company information
    @param actor - Apify Actor instance
    """
    name: str = "Google Search"
    description: str = """
    Searches google for a given query and returns the results
    """
    actor: Actor = Field(description="Apify Actor instance")
```

### Usage Example

```
search_tool = GoogleSearchTool(actor=actor)
search_results = search_tool._run("example company products")
```

Common Implementation Pattern
-----------------------------

All tools follow a similar implementation pattern:

1.   **Synchronous Wrapper**

```
def _run(self, input_param: str) -> Dict:
    """Execute synchronously by creating a new event loop"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(self._async_run(input_param))
    finally:
        loop.close()
```

1.   **Async Implementation**

```
async def _async_run(self, input_param: str) -> Dict:
    """Async implementation of the tool"""
    # Tool-specific implementation
```

Error Handling
--------------

Tools implement comprehensive error handling:

```
try:
    actor_run = await self.actor.call(actor_id="actor_id", run_input=run_input)
    if actor_run is None:
        raise RuntimeError('Actor task failed to start.')
except Exception as e:
    logging.error(f"Error in tool execution: {str(e)}")
    raise
```