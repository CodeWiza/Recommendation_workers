import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, WebsiteSearchTool
from langchain_openai import ChatOpenAI
from llama_index.program.openai import OpenAIPydanticProgram
from llama_index.core import ChatPromptTemplate
from llama_index.core.llms import ChatMessage
from llama_index.llms.openai import OpenAI
import json
from langchain_community.tools.tavily_search import TavilySearchResults


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
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)

    # Create the search agent
    agent = Agent(
        role='Supplier Researcher',
        goal='Find supplier for {topic} in specified country {topic}',
        backstory="""As a meticulous and detail-oriented individual, I have always been fascinated by the intricate web of supply chains that connect businesses and industries worldwide. With a keen analytical mind and a talent for thorough investigation, I have honed my skills in supplier research and verification to become a leading expert in the field. My dedication to detail and drive for excellence has earned me a reputation for providing the most comprehensive and trustworthy supplier networks for my clients.""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=[search_tool, web_rag_tool],
        async_execution=True
    )

    # Define the search task
    search_task = Task(
        description="As a Supplier Researcher, I am responsible for identifying and verifying reliable suppliers for a specific product in a given country. My expertise lies in conducting thorough research, analyzing market trends, and evaluating supplier credibility and quality of goods. I work closely with procurement teams to understand their specific needs and requirements, and provide tailored solutions to optimize procurement strategies and ensure seamless operations. With my expertise, companies can rest assured that their supplier network is robust, reliable, and committed to delivering high-quality goods and services.",
        expected_output="""
           Output must have 10 supplier name only in list
           Supplier name:
           Supplier_name: list of all Supplier
        """,
        agent=agent,
        tools=[search_tool, web_rag_tool]
    )

    # Create the search crew
    search_crew = Crew(
        agents=[agent],
        tasks=[search_task],
        verbose=10,
        manager_llm=llm,
        process=Process.sequential,
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



def extract_information_from_company(transcript):
    llm = OpenAI(model="gpt-3.5-turbo", api_key=os.getenv('OPEN_API_KEY'))

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


    dict_example = {
        "Supplier_details": ["Supplier 1 Name", "Supplier 2 Name", "...."],

    }

    json_example = json.dumps(dict_example)

    # Format messages using the prompt and provided transcript
    messages = prompt.format_messages(json_example=json_example, transcript=transcript)

    # Use OpenAI's llm to process the messages and extract response
    output = llm.chat(messages, response_format={"type": "json_object"}).message.content

    return output



def supplier(prompt):
    search_result = run_full_search_process(prompt)
    print(search_result)
    output = extract_information_from_company(search_result)
    output = json.loads(output)
    with open('supplier.json', 'w') as f:
        json.dump(output, f)
    return output

