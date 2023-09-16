import streamlit as st
import matplotlib.pyplot as plt


def usage_guide_page():
    st.write("### Navigating and utilizing the Dashboard effectively.")

    st.info(
        f"1. **Project Overview page**\n\n"
        f"This page gives you a clear picture of the project and its purpose. "
        f"It covers details about the dataset used, breaks down the problem "
        f"we're addressing, and explains the step-by-step the plan we've "
        f"created to successfully handle and solve the challenge at hand."
    )

    st.info(
        f"2. **Usage Guide page**\n\n"
        f"The Usage Guide page provides step-by-step instructions and helpful "
        f"tips to navigate and make the most of the features offered by "
        f"the dashboard. It ensures a smooth and effective user experience, "
        f"enabling you to fully harness the dashboard's capabilities."
    )

    st.info(
        f"3. **Project Hypothesis page**\n\n"
        f"This section serves as a concise yet comprehensive overview of "
        f"the machine learning project, outlining its anticipated results. "
        f"It holds particular value for stakeholders, business proprietors,"
        f"executives, and managers vested in making strategic "
        f"determinations predicated on the project's eventual outcomes."
    )

    st.info(
        f"4. **Cherry Leaf Visualiaser page**\n\n"
        f"On this page, you can closely examine the visual distinctions "
        f"between healthy cherry leaves and those affected by powdery "
        f"mildew infection. Discover the visual indicators that "
        f"characterize the presence of powdery mildew.\n\n"
        f"While the model's accuracy is impressive, factors such as image "
        f"quality and high contrast can occasionally influence predictions. "
        f"In cases where the model's confidence in its prediction is not "
        f"optimal, you have the opportunity to analyze the attributes of "
        f"healthy and infected leaves. This combined approach leverages both "
        f"the machine learning model and human observation, enhancing "
        f"overall accuracy."
    )

    tick_icon = "\u2713"

    st.success(
        f"**Step by step**\n\n"
        f"*To visually observe difference between average and variability "
        f"image for healthy and powdery mildew lables:*\n\n"
        f"{tick_icon} Tick the first checkbox - **Difference between"
        f" average and variability images**.\n\n"
        f"*To visually observe difference between healthy and powdery "
        f"mildew cherry leaves:*\n\n"
        f"{tick_icon} Tick the second checkbox - **Difference between "
        f"Healthy and Powdery Mildew leaves**.\n\n"
        f"*To study the features of healthy and infected cherry leaves, "
        f"use image montage feature.*\n\n"
        f"{tick_icon} Tick the last checkbox - **Image Montage**.\n"
        f"- Choose the label you want to study from the dropdown menu.\n"
        f"- Click the **Create Montage** button."
    )
