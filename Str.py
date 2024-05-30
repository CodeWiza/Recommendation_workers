import streamlit as st
import os
import pysqlite3
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
from dotenv import load_dotenv
import pycountry
from PIL import Image
from Supplier import supplier
from PIL import Image, ImageDraw, ImageOps
import json
from merge import details
from Review import details_review


import streamlit as st

# You can always call this function where ever you want



# CSS styles
css = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');

    body {{
        font-family: 'Roboto', sans-serif;
        background-color: #f7f7f7;
        color: #333;
        text-align: left;
    }}

    hr {{
        border: 0;
        height: 1px;
        background-color: #ccc;
        margin: 20px 0;
    }}

    h1 {{
        color: #5E17EB;
        font-weight: 700;
        font-size: 48px;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        animation: float 3s ease-in-out infinite;
    }}

    .stRadio {{
        display: flex;
        align-items: center;
        margin-bottom: 20px;
    }}

    .stRadio > label {{
        font-size: 20px;
        font-weight: 700;
        color: #5E17EB;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2);
        margin-right: 10px;
    }}

    .stRadio > div {{
        display: flex;
        align-items: center;
    }}

    .stRadio > div > span {{
        margin: 0 10px;
    }}

    .stMultiSelect > div > div > div > div {{
        background-color: #ecf0f1;
        border-radius: 10px;
        padding: 10px;
        color: #5E17EB;
        font-size: 18px;
        box-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
        transition: transform 0.3s ease-in-out;
    }}

    .stMultiSelect > div > div > div > div:hover {{
        transform: scale(1.05);
    }}

    .stTextInput > div > div > input {{
        font-size: 18px;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #bdc3c7;
        box-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
        transition: transform 0.3s ease-in-out;
    }}

    

    .stTextInput > div > div > input:focus {{
        transform: scale(1.05);
        outline: none;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.3);
    }}

    .stTextInput > label {{
        font-size: 20px;
        font-weight: 700;
        color: #5E17EB;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2);
    }}

    .css-q8sbsg p {{
      font-size: 32px;
      color: red;
      }}

</style>
"""


# Streamlit app
st.set_page_config(page_title="Supplier Discovery",
                   page_icon=":globe_with_meridians:", layout="wide")
st.markdown(css, unsafe_allow_html=True)


def add_logo(logo_path, width, height, radius):
    """Read, resize, and return a logo with rounded edges."""
    logo = Image.open(logo_path)
    logo = logo.resize((width, height))
    
    # Create a mask for rounded corners
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle([(0, 0), (width, height)], radius=radius, fill=255)

    # Apply the mask to the logo
    rounded_logo = ImageOps.fit(logo, (width, height), centering=(0.5, 0.5))
    rounded_logo.putalpha(mask)

    return rounded_logo

# Example usage
st.sidebar.image(add_logo(logo_path="logo.png", width=190, height=60, radius=15))
# Add logo to sidebar

# Sidebar filters
st.sidebar.title("Filters")
header_html = """
<div style='text-align: center;'>
    <h1>Supplier Discovery</h1>
    <p>Find the best suppliers for your business from around the world.</p>
</div>
"""
st.markdown(header_html, unsafe_allow_html=True)


location_filter_type_options = {
    "By Country/State": "By Country/State",
    "By Zip Code": "By Zip Code"
}
location_filter_type = st.sidebar.radio("Location Filter Type", list(
    location_filter_type_options.keys()), format_func=lambda x: location_filter_type_options[x])

data_select = []
data_select_zip = []
if location_filter_type == "By Country/State":
    # Get a list of country names
    country_names = [country.name for country in list(pycountry.countries)]
    selected_countries = st.sidebar.multiselect(
        "Select Country", country_names)
    data_select.append(selected_countries)
    if selected_countries:
        state_options = []
        for country_name in selected_countries:
            country_code = pycountry.countries.get(name=country_name).alpha_2
            states = pycountry.subdivisions.get(country_code=country_code)
            state_names = [state.name for state in states]
            state_options.extend(state_names)
        state_province = st.sidebar.multiselect(
            "Select State/Province", state_options)
        data_select.append(state_province)

elif location_filter_type == "By Zip Code":
    located_within_options = ["50 Miles", "100 Miles", "200 Miles"]
    located_within = st.sidebar.multiselect(
        "Located Within", located_within_options)
    zip_code = st.sidebar.text_input("Enter Zip Code")
    if zip_code:
        data_select_zip.append(zip_code)


search_term = st.sidebar.text_input(
    "Search for your Suppliers", placeholder="🔍 Search for your Product", key="sidebar_search_term")
search_term_main = st.text_input(
    "Search for your Suppliers", placeholder="🔍 Search for your Product", key="main_search_term")

search = None
if search_term:
    if (search_term) and (data_select or data_select_zip):
        if data_select:
            text = ', '.join(str(i[0]) for i in data_select)
            search = "I want to buy " + \
                (search_term or search_term_main) + " in " + text
        if data_select_zip:
            text = ''.join(data_select_zip)
            search = "I want to buy " + \
                (search_term or search_term_main) + " of Pin-code " + text
if search_term_main:
    search = search_term_main


def display_social_links(social_handles):
    if social_handles:
        links = [f"[{platform.capitalize()}]({link})" for platform,
                 link in social_handles.items() if link]
        if links:
            st.markdown(" | ".join(links), unsafe_allow_html=True)

# Caching mechanism


@st.cache_data
def process_data(search):
    supplier_data = supplier(search)
    return supplier_data


@st.cache_data
def processing(search):
    processed_data = details(str(search))
    return processed_data

@st.cache_data
def processing_review_data(search):
    review_data = details_review(str(search))
    return review_data


# Spinner implementation
if search:
    supplier_data = process_data(search)
else:
    supplier_data = None


if supplier_data:
    selected_suppliers = []
    with st.expander(label="Select your supplier", expanded=True):
        for key, supplier in enumerate(supplier_data["Supplier_details"]):
            supplier_container = st.container()
            cols = st.columns([1, 3, 1.5, 1.5, 2])
            with supplier_container:
                checkbox_value = cols[0].checkbox(
                    "", key=key, value=key in selected_suppliers)
                cols[1].write(supplier)
                # invitation_button = cols[4].button("Send Invite", key=f"invite_{key}")
                if checkbox_value:
                    selected_suppliers.append(key)
                elif key in selected_suppliers:
                    selected_suppliers.remove(key)
                # if invitation_button:
                   # st.success(f"Invitation sent to {supplier['Company_name']}")'''
            # st.write("---")  # Add a horizontal line after each supplier

    selected_supplier_names = [processing(str(supplier_data["Supplier_details"][index])) for index in selected_suppliers]
    # selected_supplier_names = [str(supplier_data["Supplier_details"][index]) for index in selected_suppliers]
    if selected_supplier_names:
        st.subheader("Selected Suppliers")
        for supplier in selected_supplier_names:
            company_details = supplier['Company_details']
            financial_details = supplier['Financial_details']
            board_member_details = supplier['Board_Member_details']
            col1, col2 = st.columns(2)

            with col2.expander(label="**Company Summary**", expanded=True):
                st.write(company_details['Company_summary'])

            with col1.expander(label="**Company Details**", expanded=True):
                st.write(f"**Company Name:** {company_details['Company']}")
                st.write(f"**Website:** {company_details['Website']}")
                st.write(
                    f"**Products:** {', '.join(company_details['Products'])}")
                st.write(
                    f"**Regions:** {', '.join(company_details['Regions'])}")
                st.write(f"**Phone:** {company_details['Phone_details']}")
                st.write(f"**Email:** {company_details['Email']}")

            col1, col2 = st.columns(2)

            with col1:
                with st.expander("Financial Details"):
                    for key, value in financial_details.items():
                        if key != "Company_name" and value:
                            st.write(
                                f"**{key.replace('_', ' ').title()}:** {value}")

            with col2:
                with st.expander("Board Member Details"):
                    st.write(
                        f"**Company Name:** {board_member_details['Company_name']}")
                    st.write("**Board Members:**")
                    for member in board_member_details['Board_Members']:
                        st.write(f"- {member}")

            with st.expander("### News and Additional Information"):
                news_items = board_member_details['News']
                additional_info = board_member_details['Additional_information']
                st.markdown("#### News")
                for item in news_items:
                    st.markdown(f"- {item}")

                st.markdown("#### Additional Information")
                for item in additional_info:
                    st.markdown(f"- {item}")

            with st.expander("Social Media Handles"):
                social_handles = supplier['Social_handles']['Social_details']
                display_social_links(social_handles)

            

            with st.expander("Company Review"):
                with st.form("review_form"):
                    name_supplier = company_details['Company']
                    st.write(F"Check Review for {name_supplier}")
                    submitted = st.form_submit_button("Review")
                    if submitted:
                        output = processing_review_data(name_supplier)
                        st.markdown("#### Positive Reviews")
                        for review in output["Reviews"]["positive_reviews"]:
                                st.markdown(f"##### {review['customer_name']} - {review['Rating']}")
                                st.write(review["review_summary"])
                                st.caption(f"Source: {review['website_source']}")
                        st.markdown("---")

                        st.markdown("#### Critical Reviews")
                        for review in output["Reviews"]["critical_reviews"]:
                                st.markdown(f"##### {review['customer_name']} - {review['Rating']}")
                                st.write(review["review_summary"])
                                st.caption(f"Source: {review['website_source']}")

                        st.empty()
            with st.expander("Scores"):
                col1, col2 = st.columns([2, 2])
                with col1:
                    st.markdown('#### Financial Score :- 73%')
                with col2:
                    st.markdown('#### Product Review Score :- 8/10')

        st.write("------------------------------")


# Footer section
footer_html = """
<div style='text-align: center; margin-top: 50px;'>
    <p>Powered by Supplier Discovery</p>
    <p>Stay connected with us:</p>
    <a href='https://www.facebook.com' target='_blank'><img src='https://cdn.jsdelivr.net/npm/simple-icons@3.0.1/icons/facebook.svg' width='32' height='32'></a>
    <a href='https://www.twitter.com' target='_blank'><img src='https://cdn.jsdelivr.net/npm/simple-icons@3.0.1/icons/twitter.svg' width='32' height='32'></a>
    <a href='https://www.instagram.com' target='_blank'><img src='https://cdn.jsdelivr.net/npm/simple-icons@3.0.1/icons/instagram.svg' width='32' height='32'></a>
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)
