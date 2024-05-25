from concurrent.futures import ProcessPoolExecutor
import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, WebsiteSearchTool
from langchain_openai import ChatOpenAI
from llama_index.program.openai import OpenAIPydanticProgram
from llama_index.core import ChatPromptTemplate
from llama_index.core.llms import ChatMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from llama_index.llms.openai import OpenAI
import json
from Supplier import supplier


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
    llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0)

    # Create the search agent

    merged_agent = Agent(
        role='Supplier Detailed Researcher',
        goal='Gather comprehensive information about supplier companies, including company summaries, website URLs, at least 10 product offerings, all regions of operation, at least 5 phone numbers with country codes, and 5 email addresses. Analyze each company financial statements and extract key numerical metrics related to revenue, profitability, cash flow, leverage, efficiency, and valuation ratios. Additionally, identify and document the board members for each supplier company. for {topic}',
        backstory="""Here is the merged backstory combining the provided narratives:
    As a seasoned researcher with exceptional analytical skills and a relentless curiosity, you are driven to uncover comprehensive insights about potential supplier companies. With a keen eye for detail and unparalleled research prowess, you delve deep into supplier networks, meticulously gathering information on company histories, websites, product offerings, regions of operation, and verified contact details. Your dedication to thoroughness ensures that your organization has all the necessary information for informed procurement decisions.
    Moreover, your expertise extends to the realm of financial analysis, where you possess an innate ability to decipher the intricate language of numbers that underlie a company's financial performance. You navigate through complex financial statements, extracting invaluable insights into revenue growth, profitability, cash flow, leverage, efficiency, and valuation ratios. With an analytical mindset and a deep understanding of financial metrics, you transform rows of data into coherent narratives that guide strategic decision-making and drive sustainable growth.
    Furthermore, your commitment to uncovering the truth extends to the realm of corporate governance. As a seasoned researcher with a deep understanding of leadership structures, you specialize in identifying and verifying the board members of supplier companies. With an unwavering commitment to accuracy and comprehensiveness, you meticulously sift through corporate filings, official websites, and reputable sources to compile detailed profiles of each board member, providing your organization with invaluable insights into the governance dynamics and leadership of potential suppliers.
    Your relentless pursuit of excellence, coupled with your exceptional research skills and analytical prowess, makes you an invaluable asset in fostering efficient and effective procurement processes, guiding strategic decision-making, and building robust and trustworthy business relationships.""",
        verbose=True,
        llm=llm,
        tools=[search_tool, web_rag_tool],)

    # Define the search task
    merged_task = Task(
        description="As a Comprehensive Supplier Researcher, you gather extensive details on potential suppliers, including company overviews, websites, product offerings, operational regions, and verified contact information. You conduct in-depth financial analysis by evaluating key metrics from financial statements to assess companies' financial health, efficiency, and profitability. Additionally, you identify and document the board members of each supplier company, shedding light on their leadership and governance structures. Your thorough research supports informed decision-making and strengthens supplier relationships within the organization.",
        expected_output="""Comprehensive supplier research report including:
        - Detailed profiles (company summaries, historical background, current status, key achievements)
        - List of supplier website URLs (direct links, subdomains, specific pages)
        - Analysis of product offerings (range, specifications, unique selling propositions)
        - Geographical locations of supplier operations (countries, states, specific regions)
        - Verified contact information (phone numbers(atleast 2), emails(atleast 2), physical addresses)
        
        Revenue:
    Total Revenue: [AMOUNT]
    Revenue Growth Rate (YoY): [PERCENTAGE]
    Industry Benchmark Revenue Growth: [PERCENTAGE]
    
    Profitability:
    Gross Profit: [AMOUNT]
    Gross Profit Margin: [PERCENTAGE]
    Operating Profit (EBIT): [AMOUNT]
    Operating Profit Margin: [PERCENTAGE]
    Net Profit: [AMOUNT]
    Net Profit Margin: [PERCENTAGE]
    
    Cash Flow:
    Operating Cash Flow: [AMOUNT]
    Free Cash Flow: [AMOUNT]
    
    Leverage:
    Debt-to-Equity Ratio: [RATIO]
    
    Efficiency:
    Return on Equity (ROE): [PERCENTAGE]
    Return on Assets (ROA): [PERCENTAGE]
    
    Valuation:
    Earnings Per Share (EPS): [AMOUNT]
    Price-to-Earnings (P/E) Ratio: [RATIO]
    
    Note: All amounts should be in the company's reporting currency, and relevant time periods (e.g., quarterly, annual) should be specified.

    Company Name
    List of Board Members
      Name
      Title/Position
      News
      Position
      Any additional relevant information or insights gleaned from your research

        """,
        agent=merged_agent,  # Assuming merged_agent is the combined agent from the previous example
        # Assuming these tools are shared among the tasks
        tools=[search_tool, web_rag_tool]
    )

    # Create the search crew
    search_crew = Crew(
        agents=[merged_agent],
        tasks=[merged_task],
        verbose=10,
        manager_llm=llm,
        process=Process.sequential,
        memory=True,
        full_output=True
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
                "You are an expert assistant for summarizing and extracting insights from transcripts.\n"
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

    dict_example = {"Company_details":{
            "Company": "Supplier 1",
            "Company_summary": "Summary about supplier",
            "Website": "https://supplier1.com",
            "Products": ["Product A", "Product B", "..."],
            "Regions": ["Region 1", "Region 2", "...."],
            "Phone_details": "Phone number details",
            "Email": "Email details"
        },}
    json_example = json.dumps(dict_example)
    messages = prompt.format_messages(json_example=json_example, transcript=transcript)
    output = llm.chat(messages, response_format={"type": "json_object"}).message.content
    return output

def extract_financial_from_company(transcript):
    llm = OpenAI(model="gpt-3.5-turbo", api_key=os.getenv('OPEN_API_KEY'))

    prompt = ChatPromptTemplate(
    message_templates=[
        ChatMessage(
            role="system",
            content=(
                "You are an expert assistant for summarizing and extracting insights from transcripts.\n"
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

    dict_example = {"Financial_details":{
        "Company_name": "",
        "Total_Revenue":"",
        "Revenue_Growth_Rate_YoY":"",
        "Industry_Benchmark_Revenue_Growth":"",
        "Gross_Profit":"",
        "Gross_Profit_Margin":"",
        "Operating_Profit_EBIT":"",
        "Operating_Profit_Margin":"",
        "Net_Profit":"",
        "Net_Profit_Margin":"",
        "Operating_Cash_Flow":"",
        "Free_Cash_Flow":"",
        "Debt_to_Equity_Ratio":"",
        "Return_on_Equity_ROE":"",
        "Return_on_Assets_ROA":"",
        "Earnings_Per_Share_EPS":"",
        "Price_to_Earnings_PE_Ratio":""},}

    json_example = json.dumps(dict_example)
    messages = prompt.format_messages(json_example=json_example, transcript=transcript)
    output = llm.chat(messages, response_format={"type": "json_object"}).message.content
    return output


def extract_Board_from_company(transcript):
    llm = OpenAI(model="gpt-3.5-turbo", api_key=os.getenv('OPEN_API_KEY'))

    prompt = ChatPromptTemplate(
    message_templates=[
        ChatMessage(
            role="system",
            content=(
                "You are an expert assistant for summarizing and extracting insights from transcripts.\n"
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

    dict_example = {"Board_Member_details":{
        "Company_name": "",
        "Board_Members": ["Name, News, Position of Board member 1", "Name, News, Position of Board member 2", "...."],
        "News":"News about company(atleast 5 points)",
        "Additional_information": "Information about board member (atleast 5 points)"},}

    json_example = json.dumps(dict_example)
    messages = prompt.format_messages(json_example=json_example, transcript=transcript)
    output = llm.chat(messages, response_format={"type": "json_object"}).message.content
    return output


def supplier_details(prompt):
    search_results = run_full_search_process(prompt)
    company_results = extract_information_from_company(search_results)
    financial_results = extract_financial_from_company(search_results)
    board_results = extract_Board_from_company(search_results)
    company_results = json.loads(company_results)
    financial_results = json.loads(financial_results)
    board_results = json.loads(board_results)
    merged_json = {**company_results, **financial_results, **board_results}
    return merged_json


def details(numb):
    numbers = numb['Supplier_details']
    results = []
    for data in numbers:
        out = supplier_details(data)
        results.append(out)
    with open('detained3.json', 'w') as f:
        json.dump(results, f, indent=4)
    return results








