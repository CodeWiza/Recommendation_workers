import streamlit as st
import os
import pysqlite3
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, WebsiteSearchTool
import json
import asyncio
import re
from bs4 import BeautifulSoup
import re
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.chat_models import ChatCohere
from langchain_openai import ChatOpenAI
import streamlit as st
from loguru import logger as log
from llama_index.program.openai import OpenAIPydanticProgram
from llama_index.core import ChatPromptTemplate
from llama_index.core.llms import ChatMessage
from llama_index.llms.openai import OpenAI
import json
load_dotenv()


def set_api_keys():
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv('OPEN_API_KEY')
    os.environ["TAVILY_API_KEY"] = os.getenv('TAVILY_API_KEY')
    os.environ["SERPER_API_KEY"] = os.getenv('SERPER_API_KEY')
    os.environ["CLARIFAI_PAT"] = os.getenv('CLARIFAI_PAT')


set_api_keys()

search_tool = SerperDevTool()


def create_search_agent():
    web_rag_tool = WebsiteSearchTool()
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    return Agent(
        role='Researcher',
        goal='Conduct foundational research of product {topic} and specified country {topic} and company must be public-government sector if list not full fill then search top private company also and strict find Review, email, phone-number(should be integer)',
        backstory="""You a top skilled researcher and can easily find company name of product and list all of them in form of list and return JSON file as having heading like 1.Introducation of product 2.Overview of product 3. Top supplier in India according to number user Asked and return Company_name, Company_websitelink, company_country ,summary,product offerings(company all product details),Reviews(Search on top 5 Social media website and list them with name of each about review), Email(atleast 2 emails-ids), Phone_number about company(compulsory should be integer) 4.Conculsion and 5.Reference. Very Important details needed => Reviews, Email(atleast 2 emails-ids), Phone_number about company """,
        verbose=True,
        allow_delegation=False,
        llm = llm,
        tools=[search_tool, web_rag_tool],
        async_execution=True
    )


def run_search_task(agent, result):
    web_rag_tool = WebsiteSearchTool()
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    Search_task = Task(
        description=("Gather relevant data of product and company."),
        expected_output='Gather relevant data and list all of them in form of list and return JSON file as having heading like these strict 1.Introducation of product 2.Overview of product 3.Top suppliers according to number user Asked and return Company_name, Company_websitelink, company_country ,summary,product offerings(company all product details),Reviews, Email, Phone_number about company 4.Conculsion and 5.Reference . Return all points in Json type and accurate and beautiful view and heading for each json should be same no spacing noting strict write json points ',
        agent=agent,  tools=[search_tool, web_rag_tool]
    )
    Search_crew = Crew(
        agents=[agent],
        tasks=[Search_task],
        verbose=10,
        manager_llm = llm,
        process=Process.sequential,
        full_output=True
    )
    return Search_crew.kickoff(inputs={'topic': result})


def run_financial_analysis(topic):
    web_rag_tool = WebsiteSearchTool()
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    finance_Analyser = Agent(
        role='Researcher',
        goal='Conduct foundational research of company {topicd} and find lots of fincical data of Compnay_name,Financial_highlights,Positives,Key Concerns,referenceces and gather so much data from atleast 10 financial website of realted company. extract all Company_name => {topicd} and search data one-by-one and store into list',
        backstory="""You a top skilled expert resaerch and can easily find all financial data and neccesary need all data Compnay_name,Financial_highlights,Positives,Key Concern,referenceces and gather so much data from of companies one-by-one and then store into list and then go for another company """,
        verbose=True,
        llm=llm,
        allow_delegation=False,
        tools=[search_tool, web_rag_tool],
        output_file='2.json'
    )

    finance_writing_analyser = Task(description=('Analyze the data of company Compnay_name,Financial_highlights,Positives,Key Concerns,referenceces with link '),
                                expected_output='Analyze the data of company Compnay_name,Financial_highlights,Positives,Key Concerns,referenceces with link ',  agent=finance_Analyser,
                                tools=[search_tool, web_rag_tool])

    finance_summary_crew = Crew(
        agents=[finance_Analyser],
        tasks=[finance_writing_analyser],
        verbose=10,
        manager_llm=llm,
        process=Process.sequential,
        full_output=True
    )

    return finance_summary_crew.kickoff(inputs={'topicd': topic})


def run_writer_task(topic):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    web_rag_tool = WebsiteSearchTool()
    finance_writer = Agent(
        role='Writer',
        goal='Conduct foundational research of {topicd} for each part of news for Legal issues about a supplier,board members of the supplier,issues with the board member,financial wrong of the company,Labour Strike,Refernce and gather all data company_name => {topicd}',
        backstory="""You a top skilled expert resaerch and can easily find Legal issues about a supplier,board members of the supplier,issues with the board member, financial wrong doing of the company,Labour Strike,Refernce with link you search only news website""",
        verbose=True,
        allow_delegation=False,
        llm = llm,
        tools=[search_tool, web_rag_tool],
        output_file='3.json'
    )

    finance_writing_task = Task(
        description=('Analyze the data of company for Legal-issuee,Board-Member,Issues Board-member,Financial-Wrong,Labour-Strike,Refernce'),
        expected_output='Gather relevant data for Legal-issues,Board-Member,Issues Board-member,Financial-Wrong,Labour-Strik,Refernce with link',
        output_file=f"3.json", agent=finance_writer
    )

    finance_crew = Crew(
        agents=[finance_writer],
        tasks=[finance_writing_task],
        verbose=10,
        manager_llm = llm,
        process=Process.sequential,
        full_output=True
    )

    return finance_crew.kickoff(inputs={'topicd': topic})


def extract_information_from_company(transcript):
    llm = OpenAI(model="gpt-3.5-turbo-1106", api_key=os.getenv('OPEN_API_KEY'))

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
        "Introduction of Product": "A brief introduction",
        "Overview of Product": "Overview of Product",
        "Top Suppliers": ["Company Name", "Website", "Country", "Summary", "Product Offerings", "Reviews", "Emails", "Phone Numbers"],
        "Conclusion": "A conclusion summarizing the reliability of the listed companies",
        "References": "A list of references related to the information provided. and strict must be link",
    }

    json_example = json.dumps(dict_example)

    # Format messages using the prompt and provided transcript
    messages = prompt.format_messages(json_example=json_example, transcript=transcript)

    # Use OpenAI's llm to process the messages and extract response
    output = llm.chat(messages, response_format={"type": "json_object"}).message.content

    return output


def extract_company_name(transcript):
    llm = OpenAI(model="gpt-3.5-turbo-1106", api_key=os.getenv('OPEN_API_KEY'))

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
        "Companies": ['Company_name 1','Company_name2'],
        "Website_link": ['Website_company_name 1', 'Website_company_name 2']
    }

    json_example = json.dumps(dict_example)

    # Format messages using the prompt and provided transcript
    messages = prompt.format_messages(json_example=json_example, transcript=transcript)

    # Use OpenAI's llm to process the messages and extract response
    output = llm.chat(messages, response_format={"type": "json_object"}).message.content

    return output

def extract_finance_information_from_text(transcript):
    llm = OpenAI(model="gpt-3.5-turbo-1106", api_key=os.getenv('OPEN_API_KEY'))

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
        "Company" : "Company information",
        "Financial Information" : "atleast 5 big article for Financial_highlights Company ",
        "Positives": "atleast 5 big article points for Positive of company ",
        "Key Concerns": "atleast 5 key points of company ",
        "References": "all link realted to companyand strict must be link ",
    }

    json_example = json.dumps(dict_example)

    # Format messages using the prompt and provided transcript
    messages = prompt.format_messages(json_example=json_example, transcript=transcript)

    # Use OpenAI's llm to process the messages and extract response
    output = llm.chat(messages, response_format={"type": "json_object"}).message.content

    return output

def extract_news_information_from_text(transcript):
    llm = OpenAI(model="gpt-3.5-turbo-1106", api_key=os.getenv('OPEN_API_KEY'))

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
        "Company" : "Company 1",
        "Legal Issues" : "atleast 5 big article points for Legal Issues Company ",
        "Board Members": "Board Members of company ",
        "Issues with Board Members": "atleast 5 key points Issues with Board Members of company ", 
        "Financial Wrongdoing": "Financial Wrongdoing of company ",
        "Labour Strikes": "atleast 5 key points Labour Strikes of company ",
        "References": "all link realted to company ",
    }

    json_example = json.dumps(dict_example)

    # Format messages using the prompt and provided transcript
    messages = prompt.format_messages(json_example=json_example, transcript=transcript)

    # Use OpenAI's llm to process the messages and extract response
    output = llm.chat(messages, response_format={"type": "json_object"}).message.content

    return output


def extractd(transcript):
    llm = OpenAI(model="gpt-3.5-turbo-1106", api_key=os.getenv('OPEN_API_KEY'))

    prompt = ChatPromptTemplate(
    message_templates=[
        ChatMessage(
            role="system",
            content=(
                "You are an expert assitant for data like company_summary, financial_summary and news_Summary for each company can You reccomand me one company which one I should go for it to buy material.\n"
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
        "Company" : "Company name out of all company",
        "Why the recommendation" : "Reccomdation of company reason why",
        "Positives": "Postivity of company atleast 5 points ",
        "Negative": "negativiy of company atleast 5 points ",
    }

    json_example = json.dumps(dict_example)

    # Format messages using the prompt and provided transcript
    messages = prompt.format_messages(json_example=json_example, transcript=transcript)

    # Use OpenAI's llm to process the messages and extract response
    output = llm.chat(messages, response_format={"type": "json_object"}).message.content

    return output

def display_company_recommendation(company_data):
    # Display company name
    if "Company" in company_data:
        st.subheader("Company")
        st.write(company_data["Company"])

    # Display why the recommendation
    if "Why the recommendation" in company_data:
        st.subheader("Why the recommendation")
        st.write(company_data["Why the recommendation"])

    # Display positives
    if "Positives" in company_data:
        st.subheader("Positives")
        st.write(company_data["Positives"])

    # Display negatives
    if "Negative" in company_data:
        st.subheader("Negative")
        st.write(company_data["Negative"])


def display_product_info(product_data):
    # Display introduction of product
    if "Introduction of Product" in product_data:
        st.subheader("Introduction of Product")
        st.write(product_data["Introduction of Product"])

    # Display overview of product
    if "Overview of Product" in product_data:
        st.subheader("Overview of Product")
        st.write(product_data["Overview of Product"])

    # Display top suppliers
    if "Top Suppliers" in product_data:
        st.subheader("Top Suppliers")
        suppliers = product_data["Top Suppliers"]
        for supplier in suppliers:
            st.write(f"**Company Name:** {supplier['Company Name']}")
            st.write(f"**Website:** [{supplier['Website']}]({supplier['Website']})")
            st.write(f"**Country:** {supplier['Country']}")
            st.write(f"**Summary:** {supplier['Summary']}")
            st.write(f"**Product Offerings:** {supplier['Product Offerings']}")
            st.write(f"**Reviews:** {supplier['Reviews']}")
            st.write(f"**Emails:** {supplier['Emails']}")
            st.write(f"**Phone Numbers:** {supplier['Phone Numbers']}")
            st.write("---")

    # Display conclusion
    if "Conclusion" in product_data:
        st.subheader("Conclusion")
        st.write(product_data["Conclusion"])

    # Display references
    if "References" in product_data:
        st.subheader("References")
        st.write(f"[References]({product_data['References']})")


def display_company_finance_details(company_json):
    # Parse the JSON string into a Python dictionary
    company_data = json.loads(company_json)

    # Display company name
    if "Company" in company_data:
        st.subheader("Company Name")
        st.write(company_data["Company"])

    # Display financial information
    if "Financial Information" in company_data:
        st.subheader("Financial Information")
        financial_info = company_data["Financial Information"]
        st.json(financial_info)

    # Display positives
    if "Positives" in company_data:
        st.subheader("Positives")
        positives = company_data["Positives"]
        st.json(positives)

    # Display key concerns
    if "Key Concerns" in company_data:
        st.subheader("Key Concerns")
        key_concerns = company_data["Key Concerns"]
        st.json(key_concerns)

    # Display references
    if "References" in company_data:
        st.subheader("References")
        references = company_data["References"]
        st.json(references)


def display_company_news_details(company_json):
    # Parse the JSON string into a dictionary
    company_data = json.loads(company_json)

    # Display company name
    if "Company" in company_data:
        st.subheader("Company Name")
        st.write(company_data["Company"])

    # Display legal issues
    if "Legal Issues" in company_data:
        st.subheader("Legal Issues")
        st.write(company_data["Legal Issues"])

    # Display board members
    if "Board Members" in company_data:
        st.subheader("Board Members")
        board_members = company_data["Board Members"]
        if isinstance(board_members, list) and len(board_members) > 0:
            for member in board_members:
                st.write(f"- {member}")
        else:
            st.write("No specific board members mentioned.")

    # Display issues with board members
    if "Issues with Board Members" in company_data:
        st.subheader("Issues with Board Members")
        st.write(company_data["Issues with Board Members"])

    # Display financial wrongdoing
    if "Financial Wrongdoing" in company_data:
        st.subheader("Financial Wrongdoing")
        st.write(company_data["Financial Wrongdoing"])

    # Display labour strikes
    if "Labour Strikes" in company_data:
        st.subheader("Labour Strikes")
        st.write(company_data["Labour Strikes"])

    # Display references
    if "References" in company_data:
        st.subheader("References")
        references = company_data["References"]
        if isinstance(references, list):
            for ref in references:
                if isinstance(ref, dict) and "title" in ref and "link" in ref:
                    st.write(f"- [{ref['title']}]({ref['link']})")
                elif isinstance(ref, str):
                    st.write(f"- {ref}")
        else:
            st.write("No references available.")


search_agent = create_search_agent()
st.set_page_config(
    page_title="Supplier Discovery App",
    layout="wide",
    initial_sidebar_state="expanded",  # Optional: expand the sidebar initially  # Set the theme to light
)
logo_path = "logo.png"
def round_image(image_path, width):
    st.markdown(
        f'<style>img.rounded{{border-radius:50%;}}</style>',
        unsafe_allow_html=True,
    )
    st.image(image_path, width=width, output_format='PNG')



# Display rounded logo image
round_image(logo_path, width=180)



# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

recco = []
st.markdown(
    f'<p style="font-size: 60px; font-weight: bold; text-align: center; color: #5F16EB;">Supplier Discovery</p>',
    unsafe_allow_html=True
)
# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)

    # Display assistant response in chat message container
    with st.spinner("Fetching details..."):
        search_result = run_search_task(search_agent, prompt)
        recco.append(search_result)
        data = extract_information_from_company(search_result)
        print("1-----------------------------------------------------------------------------")
        st.header('Details')
        data = json.loads(data)
        print(data)
        display_product_info(data)
        out = extract_company_name(data)
    st.write("--------------------------------------------------------------------------------------------")
    # Run financial analysis
    with st.spinner("Running financial analysis..."):
        data_finance = []
        print(out)
        out = json.loads(out)
        companies = out.get('Companies', [])
        for company_name in companies:
            print(company_name)
            data_fine = run_financial_analysis(company_name)
            data = extract_finance_information_from_text(data_fine)
            data_finance.append(data)
        recco.append(data_finance)
        print(data_finance)
        st.header('Financial Details')
        for idx, json_data in enumerate(data_finance):
            display_company_finance_details(json_data)
        print("2-----------------------------------------------------------------------------")
        #print(data)
    st.write("--------------------------------------------------------------------------------------------")

    # Run news summary task
    with st.spinner("Getting news details..."):
        news_summa = []
        companies = out.get('Companies', [])
        for company_name in companies:
            data_fine = run_writer_task(company_name)
            data = extract_news_information_from_text(data_fine)
            news_summa.append(data)
        recco.append(news_summa)
        print(news_summa)
        st.header('News Details')
        for idx, json_data in enumerate(news_summa):
            display_company_news_details(json_data)
    st.write("--------------------------------------------------------------------------------------------")

    with st.spinner("Running Recommdation Agent..."):
        out = extractd(recco)
        if out:
            print(out)
            st.header('Recommendation')
            out = json.loads(out)
            display_company_recommendation(out)
    # Add assistant response to chat histor
