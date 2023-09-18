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

    st.info(
        f"5. **Powdery Mildew Detection page**\n\n"
        f"This page allows you to upload an image of a Cherry leaf and predict"
        f" whether it has powdery mildew disease or it is healthy. To use "
        f"this feature, click on the **Browse files** button and select an "
        f"image from your local machine. Once Upload is finished the page will"
        f" display the prediction result with a confidence score. You can"
        f" also view predicted probabilities for the input image across "
        f"different classes."
    )

    st.success(
        f"**Step by step**\n\n"
        f"1. Open the Streamlit dashboard in a web browser.\n"
        f"2. Navigate to the *Powdery Mildew Detection* page by clicking on "
        f"the corresponding tab in the left sidebar menu.\n"
        f"3. Click on the **Browse files** button and select an image or "
        f"more from your local machine and upload it without any "
        f"complications. You can use the drag and drop feature aswell.\n"
        f"4. Predictions will appear below. Read the prediction result or "
        f"study the features and metrics as required. To read the prediction "
        f"result, the user should look for the predicted class and the "
        f"corresponding confidence score. For example, if the predicted class "
        f"is **powdery mildew** and the confidence score is 0.85, it means "
        f"that the model is 85 % confident that the input image has powdery "
        f"mildew disease.\n"
        f"5. If needed, you can scroll through the predictions and check where"
        f" the model was uncertain and make a correct decision yourself.\n"
        f"6. Download a prediction report at the bottom of the page."
    )

    st.info(
        f"6. *ML Performance Metrics:* This page provides the evaluation "
        f"metrics of the machine learning model used in the project. The user "
        f"can view the confusion matrix, precision, recall, and F1 score of "
        f"the model. The user can also study the metrics to understand the "
        f"performance of the model. It is primarily intended for technical "
        f"staff members who are responsible for building and refining the ML "
        f"model, but may also be relevant for other stakeholders who are "
        f"interested in understanding the technical performance of project."
    )
