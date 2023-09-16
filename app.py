import streamlit as st

from app_pages.multipage import MultiPage
from app_pages.project_overview_page import project_overview_page
from app_pages.usage_guide_page import usage_guide_page
from app_pages.leaf_visualizer_page import leaf_visualizer_page

app = MultiPage(app_name="Mildew Detection in Cherry Leaves")
app.add_page("Project Overview", project_overview_page)
app.add_page("Usage Guide", usage_guide_page)
app.add_page("Cherry Leaf Visualizer", leaf_visualizer_page)

app.run()
