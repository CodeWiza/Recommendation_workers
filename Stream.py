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
from streamlit_option_menu import option_menu
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
web_rag_tool = WebsiteSearchTool()


def create_search_agent():

    return Agent(
        role='Researcher',
        goal='Conduct foundational research of product {topic} and specified country {topic} and company must be public-government sector if list not full fill then search top private company also and strict find Review, email, phone-number',
        backstory="""You a top skilled researcher and can easily find company name of product and list all of them in form of list and return JSON file as having heading like 1.Introducation of product 2.Overview of product 3. Top supplier in India according to number user Asked and return Company_name, Company_websitelink, company_country ,summary,product offerings(company all product details),Reviews, Email(atleast 2 emails-ids), Phone_number about company 4.Conculsion and 5.Reference. Very Important details needed => Reviews, Email(atleast 2 emails-ids), Phone_number about company """,
        verbose=True,
        allow_delegation=False,
        tools=[search_tool, web_rag_tool],
        async_execution=True
    )


def run_search_task(agent, result):
    Search_task = Task(
        description=("Gather relevant data of product and company."),
        expected_output='Gather relevant data and list all of them in form of list and return JSON file as having heading like these strict 1.Introducation of product 2.Overview of product 3.Top suppliers according to number user Asked and return Company_name, Company_websitelink, company_country ,summary,product offerings(company all product details),Reviews, Email, Phone_number about company 4.Conculsion and 5.Reference . Return all points in Json type and accurate and beautiful view and heading for each json should be same no spacing noting strict write json points ',
        agent=agent,  tools=[search_tool, web_rag_tool]
    )
    Search_crew = Crew(
        agents=[agent],
        tasks=[Search_task],
        verbose=10,
        process=Process.sequential,
        full_output=True
    )
    return Search_crew.kickoff(inputs={'topic': result})


def run_financial_analysis(topic):
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

    finance_writer = Agent(
        role='Writer',
        goal='Conduct foundational research of little news for each part of news analyser data of 1) Legal issues about a supplier(find all issue and write in points with summary)  2) who are board members of the company (write in points) 3) Any issues with the board member (All issue and write in points) 4) financial wrong doing of the company (All financial wrong doing and summarize in points) 5) Labour Strike (All strike and combine into points) 6) Refernce  and gather less data for all points | company_name => {topicd}',
        backstory="""You a top skilled expert resaerch and can easily find 1) Legal issues about a supplier(find all issue and write in points with summary)  2) who are board members of the company (write in points) 3) Any issues with the board member (All issue and write in points) 4) financial wrong doing of the company (All financial wrong doing and summarize in points) 5) Labour Strike (All strike and combine into points) 6) Refernce for with link and U can find Easily Fast and Accurate details and u search only news data website""",
        verbose=True,
        allow_delegation=False,
        llm = llm,
        tools=[search_tool, web_rag_tool],
        output_file='3.json'
    )

    finance_writing_task = Task(
        description=('Analyze the data of company and return 1)Legal-issues 2)Board-Member 3)Issues Board-member (All issue and write in points) 4)Financial-Wrong (All financial wrong doing and summarize in points) 5)Labour-Strike (All strike and combine into points) 6)Refernce  and output should be JSON format and strict print data for all company in Json format'),
        expected_output='Gather relevant data and list all of them in form of list and return JSON only data and having heading like 1)Legal-issues 2)Board-Member 3)Issues Board-member (All issue and write in points) 4)Financial-Wrong (All financial wrong doing and summarize in points) 5)Labour-Strike (All strike and combine into points) 6)Refernce  with link of all company name of input . Return all points in Json type and accurate and beautiful view and heading for each json should be same no spacing noting',
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
        "References": "A list of references related to the information provided.",
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
        "Company" : ["Company 1", "Company 2"],
        "Financial Information" : ["atleast 5 big article for Financial_highlights Company 1","atleast 5 big article for Financial_highlights Company 2"],
        "Positives": ["atleast 5 big article points for Positive of company 1", "atleast 5 big article points for Positive of company 2"],
        "Key Concerns": ["atleast 5 key points of company 1", "atleast 5 key points of company 2"],
        "References": ["all link realted to company 1", "all link realted to company 2"]
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
        "Company" : ["Company 1", "Company 2"],
        "Legal Issues" : ["atleast 5 big article points for Legal Issues Company 1 with ","atleast 5 big article points for Legal Issues Company 2"],
        "Board Members": ["Board Members of company 1", "Board Members of company 2"],
        "Issues with Board Members": ["atleast 5 key points Issues with Board Members of company 1", "atleast 5 key points Issues with Board Members of company 2"],
        "Financial Wrongdoing": ["Financial Wrongdoing of company 1", "Financial Wrongdoing of company 2"],
        "Labour Strikes": ["atleast 5 key points Labour Strikes of company 1", "atleast 5 key points Labour Strikes of company 2"],
        "References": ["all link realted to company 1", "all link realted to company 2"]
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


def display_company_info(company_data):
    # Display company name
    if "Company" in company_data:
        st.subheader("Company")
        st.write(", ".join(company_data["Company"]))

    # Display financial information
    if "Financial Information" in company_data:
        st.subheader("Financial Information")
        for info in company_data["Financial Information"]:
            st.write(info)

    # Display positives
    if "Positives" in company_data:
        st.subheader("Positives")
        for positive in company_data["Positives"]:
            st.write("- " + positive)

    # Display key concerns
    if "Key Concerns" in company_data:
        st.subheader("Key Concerns")
        for concern in company_data["Key Concerns"]:
            st.write("- " + concern)

    # Display references
    if "References" in company_data:
        st.subheader("References")
        for ref in company_data["References"]:
            st.write(f"[{ref}]({ref})")


def display_company_details(company_data):
    # Display company name
    if "Company" in company_data:
        st.subheader("Company")
        st.write(", ".join(company_data["Company"]))

    # Display legal issues
    if "Legal Issues" in company_data:
        st.subheader("Legal Issues")
        for issue in company_data["Legal Issues"]:
            st.write("- " + issue)

    # Display board members
    if "Board Members" in company_data:
        st.subheader("Board Members")
        st.write(company_data["Board Members"])

    # Display issues with board members
    if "Issues with Board Members" in company_data:
        st.subheader("Issues with Board Members")
        for issue in company_data["Issues with Board Members"]:
            st.write("- " + issue)

    # Display financial wrongdoing
    if "Financial Wrongdoing" in company_data:
        st.subheader("Financial Wrongdoing")
        for issue in company_data["Financial Wrongdoing"]:
            st.write("- " + issue)

    # Display labour strikes
    if "Labour Strikes" in company_data:
        st.subheader("Labour Strikes")
        for strike in company_data["Labour Strikes"]:
            st.write("- " + strike)

    # Display references
    if "References" in company_data:
        st.subheader("References")
        for ref in company_data["References"]:
            st.write(f"[{ref}]({ref})") # Display as clickable link


search_agent = create_search_agent()
logo_and_title = True
if logo_and_title:
    head = st.columns([2, 3, 5, 5])
    with head[1]:
        st.image("logo.png", use_column_width=False, width=380)


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

recco = []
st.title("Recommendation-Bot")
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
        finance_sun = run_financial_analysis(out)
        recco.append(finance_sun)
        data = extract_finance_information_from_text(finance_sun)
        st.header('Financial Details')
        data = json.loads(data)
        display_company_info(data)
        print(data)
        print("2-----------------------------------------------------------------------------")
        #print(data)
    st.write("--------------------------------------------------------------------------------------------")

    # Run news summary task
    with st.spinner("Getting news details..."):
        news_sum = run_writer_task(out)
        recco.append(news_sum)
        json_data = extract_news_information_from_text(news_sum)
        data = json.loads(json_data)
        print("3-----------------------------------------------------------------------------")
        st.header('News Details')
        display_company_details(data)
    st.write("--------------------------------------------------------------------------------------------")

    with st.spinner("Running Recommdation Agent..."):
        out = extractd(recco)
        if out:
            print(out)
            st.header('Recommendation')
            out = json.loads(out)
            display_company_recommendation(out)
    # Add assistant response to chat history
