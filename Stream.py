import streamlit as st
import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, WebsiteSearchTool
import json
import asyncio
from llama_index.llms.clarifai import Clarifai
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.chat_models import ChatCohere
from langchain_openai import ChatOpenAI
load_dotenv()

def set_api_keys():
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv('OPEN_API_KEY')
    os.environ["TAVILY_API_KEY"] = os.getenv('TAVILY_API_KEY')
    os.environ["SERPER_API_KEY"] = os.getenv('SERPER_API_KEY')
    os.environ["CLARIFAI_PAT"] = os.getenv('CLARIFAI_PAT')

set_api_keys()
search_tool = TavilySearchResults()
web_rag_tool = WebsiteSearchTool()

def create_search_agent():

    return Agent(
        role='Researcher',
        goal='Conduct foundational research of product {topic} and specified country {topic} and company must be public-government sector if list not full fill then search top private company also',
        backstory="""You a top skilled researcher and can easily find company name of product and list all of them in form of list and return JSON file as having heading like 1.Introducation of product 2.Overview of product(article form information) 3. Top supplier in India according to number user Asked and return Company_name, Company_websitelink, company_country ,summary,product offerings(company all product details),Reviews(Indeed, Glassdoor and AmbitionBox), Email(atleast 5 emails-ids), Phone_number(every state mobile number) about company 4.Conculsion and 5.Reference . Return all points in Json type and accurate and beautiful view and company must be public-government sector if list not full fill then search top private company also """,
        verbose=True,
        allow_delegation=False,
        tools=[search_tool, web_rag_tool],
        output_file='1.json',
        async_execution=True
    )

def run_search_task(agent, result):
    Search_task = Task(
        description=("Gather relevant data of product and company."),
        expected_output='Gather relevant data and list all of them in form of list and return JSON file as having heading like these strict 1.Introducation of product 2.Overview of product(article form information) 3.Top suppliers according to number user Asked and return Company_name, Company_websitelink, company_country ,summary,product offerings(company all product details),Reviews(Indeed, Glassdoor and AmbitionBox), Email(atleast 5 emails-ids), Phone_number(every state mobile number) about company 4.Conculsion and 5.Reference . Return all points in Json type and accurate and beautiful view and heading for each json should be same no spacing noting strict write json points ',
        agent=agent, output_file=f"1.json", tools=[search_tool, web_rag_tool]
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
        goal='Conduct foundational research of lots of fincical analyser data of 1.Compnay_name 2. Financial_highlights 3.Positives (make summary of 5 points stleast) 4. Key Concerns (make atleast 5 points) and 5.referenceces and gather so much data for all points | company_name => {topicd}',
        backstory="""You a top skilled expert resaerch and can easily find 1.Compnay_name 2. Financial_highlights 3.Positives (make summary of 5 points stleast) 4. Key Concerns (make atleast 5 points) and 5.referenceces with link and U can find Easily Fast and Accurate details and u search only financial data website""",
        verbose=True,
        llm=llm,
        allow_delegation=False,
        tools=[search_tool, web_rag_tool],
        output_file='2.json'
    )
    
    finance_writing_analyser = Task(
        description=('Analyze the data of company and return 1.Compnay_name 2. Financial_highlights 3.Positives (make summary of 5 points stleast) 4. Key Concerns (make atleast 5 points) and 5.referenceces with link is  and output should be JSON format and strict print data for all company in Json format'),
        expected_output='Gather relevant data and list all of them in form of list and return JSON file as having heading like1.Company_name 2. Financial_highlights 3.Positives (make summary of 5 points stleast) 4. Key Concerns (make atleast 5 points) and 5.referenceces with link of all company name of input. Return all points in Json type and accurate and beautiful view and heading for each json should be same no spacing noting',
        output_file=f"2.json", agent=finance_Analyser,
        tools=[search_tool, web_rag_tool]
    )

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
    llm = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0)

    finance_writer = Agent(
        role='Writer',
        goal='Conduct foundational research of little news for each part of news analyser data of 1) Legal issues about a supplier(find all issue and write in points with summary)  2) who are board members of the company (write in points) 3) Any issues with the board member (All issue and write in points) 4) financial wrong doing of the company (All financial wrong doing and summarize in points) 5) Labour Strike (All strike and combine into points) 6) Refernce  and gather less data for all points | company_name => {topicd}',
        backstory="""You a top skilled expert resaerch and can easily find 1) Legal issues about a supplier(find all issue and write in points with summary)  2) who are board members of the company (write in points) 3) Any issues with the board member (All issue and write in points) 4) financial wrong doing of the company (All financial wrong doing and summarize in points) 5) Labour Strike (All strike and combine into points) 6) Refernce for with link and U can find Easily Fast and Accurate details and u search only news data website""", 
        verbose=True,
        allow_delegation=False,
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
        process=Process.sequential,
        full_output=True
    )
    
    return finance_crew.kickoff(inputs={'topicd': topic})

def extractd(text):
    llm_model = Clarifai(
        model_url="https://clarifai.com/openai/chat-completion/models/gpt-4-turbo")
    summary = llm_model.complete(prompt=f'''
                You have lots of data like company_summary, financial_summary and news_Summary for each company can You reccomand me one company which one 
                I should go for it to buy material and output only strict have json data only not more than
                E.g:- Output should be like in JSON also
                [
                Company_name
                [
                    "Why the recommendation":
                    "Positives":
                    "Negative" :
                ]]
                Transcription: {text}
            ''')

    summary = (str(summary))
    return summary

def display_recommendations(search_result_json):
    # Parse the JSON result
    result_dict = search_result_json

    # Display each section of the recommendation in a structured format

    if "1.Introducation of product" in result_dict and result_dict["1.Introducation of product"]:
        st.subheader("Introduction of product")
        st.write(result_dict["1.Introducation of product"])

    if "2.Overview of product" in result_dict and result_dict["2.Overview of product"]:
        st.subheader("Overview of product")
        st.write(result_dict["2.Overview of product"])

    # Display multiple top suppliers in India if available
    if "3.Top suppliers" in result_dict and result_dict["3.Top suppliers"]:
        st.subheader("Top suppliers")
        st.json(result_dict["3.Top suppliers"])

    if "4.Conculsion" in result_dict and result_dict["4.Conculsion"]:
        st.subheader("Conclusion")
        st.write(result_dict["4.Conculsion"])

    if "5.Reference" in result_dict and result_dict["5.Reference"]:
        st.subheader("Reference")
        st.write(result_dict["5.Reference"])

def display_company_information(json_data):
    # Parse JSON data into a dictionary
    data = json.loads(json_data)

    # Display Company Name if available
    if "Company_name" in data and data["Company_name"]:
        st.header("Financial Details")
        st.subheader("Company Name")
        st.write(data["Company_name"])

    # Display Financial Highlights if available
    if "Financial_highlights" in data and data["Financial_highlights"]:
        st.subheader("Financial Highlights")
        st.write(data["Financial_highlights"])

    # Display Positives if available
    if "Positives" in data and data["Positives"]:
        st.subheader("Positives")
        positives = data["Positives"]
        for key, value in positives.items():
            st.markdown(f"- {value}")

    # Display Key Concerns if available
    if "Key_Concerns" in data and data["Key_Concerns"]:
        st.subheader("Key Concerns")
        concerns = data["Key_Concerns"]
        for key, value in concerns.items():
            st.markdown(f"- {value}")

    # Display References if available
    if "References" in data and data["References"]:
        st.subheader("References")
        st.write(data["References"])

def display_company_details(json_data):
    # Parse JSON data into a dictionary
    data = json.loads(json_data)

    # Display Legal Issues if data is available
    if "Legal-issues" in data and data["Legal-issues"]:
        st.subheader("Legal Issues")
        legal_issues = data["Legal-issues"]
        for issue in legal_issues:
            st.markdown(f"- {issue}")

    # Display Board Members if data is available
    if "Board-Member" in data and data["Board-Member"]:
        st.subheader("Board Members")
        board_members = data["Board-Member"]
        for member in board_members:
            st.markdown(f"- {member}")

    # Display Issues with Board Members if data is available
    if "Issues-Board-member" in data and data["Issues-Board-member"]:
        st.subheader("Issues with Board Members")
        member_issues = data["Issues-Board-member"]
        for issue in member_issues:
            st.markdown(f"- {issue}")

    # Display Financial Wrongdoings if data is available
    if "Financial-Wrong" in data and data["Financial-Wrong"]:
        st.subheader("Financial Wrongdoings")
        financial_wrongdoings = data["Financial-Wrong"]
        for wrongdoing in financial_wrongdoings:
            st.markdown(f"- {wrongdoing}")

    # Display Labor Strikes if data is available
    if "Labour-Strike" in data and data["Labour-Strike"]:
        st.subheader("Labor Strikes")
        labor_strikes = data["Labour-Strike"]
        for strike in labor_strikes:
            st.markdown(f"- {strike}")

    # Display References if data is available
    if "Reference" in data and data["Reference"]:
        st.subheader("References")
        references = data["Reference"]
        for ref in references:
            st.write(f"- [{ref}]({ref})")  # Display as clickable link

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
        prompt = search_result["final_output"]
        response = prompt
        recco.append(response)
        json_data = json.loads(response)
        st.header('Details')
        display_recommendations(json_data)
        st.json(json_data)
    st.write("--------------------------------------------------------------------------------------------")
    # Run financial analysis
    with st.spinner("Running financial analysis..."):
        finance_sun = run_financial_analysis(response)
        recco.append(finance_sun)
        finance_sun = finance_sun["final_output"]
        display_company_information(finance_sun)
        st.json(finance_sun)
    st.write("--------------------------------------------------------------------------------------------")    

    # Run news summary task
    with st.spinner("Getting news details..."):
        news_sum = run_writer_task(response)
        recco.append(news_sum)
        news_sum = news_sum["final_output"]
        st.header('News Details')
        st.json(news_sum)
    st.write("--------------------------------------------------------------------------------------------")

    with st.spinner("Running Recommdation Agent..."):
        out = extractd(recco)
        if out:
            st.header('Recommendation')
            st.write(out)
    # Add assistant response to chat history
  
