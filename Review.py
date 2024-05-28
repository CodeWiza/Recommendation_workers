from concurrent.futures import ProcessPoolExecutor
import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, WebsiteSearchTool, ScrapeWebsiteTool, ScrapeElementFromWebsiteTool
from langchain_openai import ChatOpenAI
from llama_index.program.openai import OpenAIPydanticProgram
from llama_index.core import ChatPromptTemplate
from llama_index.core.llms import ChatMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from llama_index.llms.openai import OpenAI
import json
from Supplier import supplier
from textwrap import dedent


def run_full_search_process(prompt):
    # Load environment variables
    load_dotenv()

    os.environ["OPENAI_API_KEY"] = os.getenv('OPEN_API_KEY')
    os.environ["TAVILY_API_KEY"] = os.getenv('TAVILY_API_KEY')
    os.environ["SERPER_API_KEY"] = os.getenv('SERPER_API_KEY')
    os.environ["CLARIFAI_PAT"] = os.getenv('CLARIFAI_PAT')

    # Initialize tools and LLM
    search_tool = SerperDevTool()
    web_rag_tool = WebsiteSearchTool()
    #llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0)
    # Create the search agent

    merged_agent = Agent(
        role="Product Review collector",
        goal="gather product reviews of a company's product from various sources across the internet atleast 5 positive_reviews and 5 negative_reviews  {topic}.",
        tools=[search_tool, web_rag_tool],
        backstory=dedent("""\
        With an eye for detail and a knack for thorough research, you specialize in collecting customer reviews from a myriad of online source. your expertise lies in sifting through
        e-commerce sites, forums, social media platforms, and review aggregators (specifically excluding employee-centric platforms like AmbitionBox and Glassdoor) to gather comprehensive 
        feedback on products filtering out any potential employee reviews.Your insights will lay the groundwork for Product review collector.
        """),
        #llm=llm,     #If u want to use another llm instead of GPT-4 then remove the comment from llm=llm.
        verbous=True
      )

    # Define the search task
    merged_task = Task(
        description=dedent(f"""\
        scour the internet for a product reviwes of a company's product. gathere the maximum product reviwes.
        key Action:
        reviwes collection: Gather product reviews from various online sources, specifically excluding employee-centric platforms like AmbitionBox and Glassdoor. 
        Prioritize sources like: Company's official website, Review aggregators, Industry-specific forums and websites and Social media platforms.
        Extract information: customer Name (or Username), Review summary (Concisely summarize the main points), Ratings and Website links from where reviews were collected.
        """),
        expected_output=dedent(f"""\
        A An insightful content that includes,
        customer name, Full Review Snippet, Star ratings(numaric) and Source Website links.
        """),
        agent=merged_agent
      )

    # Create the search crew


    # Create the search crew
    search_crew = Crew(
        agents=[merged_agent],
        tasks=[merged_task],
        verbose=10,
        manager_llm=llm,
        process=Process.sequential,
        memory=True,
        full_output=True,
        embedder={
                "provider": "openai",
                "config":{
                        "model": 'text-embedding-3-small'
                }
        }
    )

    # Execute the search task
    search_result = search_crew.kickoff(inputs={'topic': prompt})

    return search_result

def extract_review_information_from_company(transcript):
    llm = OpenAI(model="gpt-3.5-turbo", api_key=os.getenv('OPENAI_API_KEY'))
    prompt = ChatPromptTemplate(
    message_templates=[
        ChatMessage(
            role="system",
            content=(
                "You are an expert assitant for summarizing and extracting insights transcripts.\n"
                "Generate a valid JSON in the following format:\n"
                "{json_example}"
            ),
        ),
        ChatMessage(
            role="user",
            content=(
                "Here is the transcript: \n"
                "------\n"
                "{transcript}\n"
                "------"
            ),
        ),
    ])

    supplier_details = {"Reviews":{
  "positive_reviews": [
    {
      "customer_name": "customer name1",
      "review_summary": "review summary",
      "Rating":"Rating stars",
      "website_source": "website source url"
    },
    {
      "customer_name": "customer name2",
      "review_summary": "review summary",
      "Rating":"Rating stars",
      "website_source": "website source url"
    }
    
  ],
  "critical_reviews": [
    {
      "customer_name": "customer name1",
      "review_summary": "review summary",
      "Rating":"Rating stars",
      "website_source": "website source url"
    },
    {
      "customer_name": "customer name2",
      "review_summary": "review summary",
      "Rating":"Rating stars",
      "website_source": "website source url"
    }
    
  ]
}}
    json_example = json.dumps(supplier_details)
    messages = prompt.format_messages(json_example=json_example, transcript=transcript)
    output = llm.chat(messages, response_format={"type": "json_object"}).message.content
    return output



def supplier_details(prompt):
    search_results = run_full_search_process(prompt)
    company_results = extract_review_information_from_company(search_results)
    social_handle = json.loads(company_results)
    merged_json = {**social_handle}
    return merged_json


def details_review(numb):
    out = supplier_details(numb)
    with open('review.json', 'w') as f:
        json.dump(out, f, indent=4)
    return out










