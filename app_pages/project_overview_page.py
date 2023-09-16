import streamlit as st
import matplotlib.pyplot as plt


def project_overview_page():
    st.write("## Project Overview")

    st.info(
        f"**Introduction**\n\n"
        f"Welcome to the Project Overview page! This project revolves around "
        f"addressing the issue of powdery mildew, a fungal disease that "
        f"commonly affects cherry trees. The current method of manually "
        f"inspecting leaves for this disease is not practical "
        f"due to the large number of cherry trees across multiple farms."
        f"Our proposed solution involves harnessing the power of machine "
        f"learning (ML). We aim to create an ML system capable of instantly "
        f"detecting the presence of powdery mildew in cherry leaf images.\n\n"
        f"**Project Dataset**\n\n"
        f"We've obtained a dataset of cherry leaf images from our client "
        f"**Farmy & Foods**, directly from their own crops. The dataset "
        f"contains +4 thousand images. The images show healthy cherry "
        f"leaves and cherry leaves that have powdery mildew."
    )

    st.write(
        f"* For more details about the dataset and how we prepared it, "
        f"check out the [README file]"
        f"(https://github.com/Edmir-Demaj/mildew-detection-in-cherry-leaves/"
        f"blob/main/README.md)"
    )

    st.success(
        f"**Business Goals:**\n"
        f"* 1 - Our client desires a comprehensive study to visually "
        f"differentiate between healthy and powdery mildew-infected "
        f"cherry leaves.\n"
        f"* 2 - Our client is keen on rapid and accurate predictions "
        f"regarding the health status of cherry trees.\n"
        f"* 3 - The client expects a dashboard solution that meets "
        f"the above-mentioned requirements.\n"
    )

    st.info(
        f"**Objectives**\n\n"
        f"* Develop an ML system capable of accurately determining "
        f"whether a given cherry leaf is healthy or infected with "
        f"powdery mildew, based solely on an image.\n"
        f"* Enhance the efficiency of powdery mildew detection across "
        f"multiple farms by providing a scalable and instantaneous solution.\n"
    )

    st.info(
        f"**Project Steps**\n\n"
        f"1. Gather a dataset of cherry leaf images generously supplied "
        f"by Farmy & Foods.\n"
        f"2. Prepare the dataset through processes like cleaning, "
        f"resizing, and normalization, optimizing it for ML algorithms.\n"
        f"3. Implement image augmentation techniques to enrich the "
        f"training dataset, ultimately enhancing the model's performance.\n"
        f"4. Develop a robust ML model utilizing supervised learning "
        f", such as convolutional neural networks (CNNs), to classify "
        f"cherry leaves as either healthy or powdery mildew-infected.\n"
        f"5. Train the model using the preprocessed dataset and validate "
        f"its effectiveness using an independent test dataset.\n"
        f"6. Deploy the trained model onto a dynamic dashboard that "
        f"showcases its predictions, adhering to our client's requirements.\n"
        f"7. Undertake a comprehensive study to distinguish between healthy "
        f"and powdery mildew-infected cherry leaves. Utilize the deployed "
        f"model to predict the health status of cherry trees.\n"
        f"8. Present potential outcomes and rationales to our client, "
        f"facilitating their decision-making process.\n"
    )
