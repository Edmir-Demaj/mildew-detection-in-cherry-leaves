import streamlit as st

# Load pages scripts
from app_pages.multipage import MultiPage
from app_pages.project_overview_page import project_overview_page
from app_pages.usage_guide_page import usage_guide_page
from app_pages.project_hypothesis import project_hypothesis_page
from app_pages.leaf_visualizer_page import leaf_visualizer_page
from app_pages.powdery_mildew_detection_page import (
    powdery_mildew_detection_page
)
from app_pages.ml_performance_page import ml_performance_page


app = MultiPage(app_name="Mildew Detection in Cherry Leaves")

# Add pages to the streamlit dashboard
app.add_page("Project Overview", project_overview_page)
app.add_page("Usage Guide", usage_guide_page)
app.add_page("Project Hypothesis", project_hypothesis_page)
app.add_page("Cherry Leaf Visualizer", leaf_visualizer_page)
app.add_page("Powdery Mildew Detection", powdery_mildew_detection_page)
app.add_page("ML Performance Metrics", ml_performance_page)

app.run()
