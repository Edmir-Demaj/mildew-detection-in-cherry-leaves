import streamlit as st

from app_pages.multipage import MultiPage
from app_pages.project_overview_page import project_overview_page

app = MultiPage(app_name="Mildew Detection in Cherry Leaves")
app.add_page("Project Overview", project_overview_page)

app.run()
