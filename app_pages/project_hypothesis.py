import streamlit as st


def project_hypothesis_page():
    st.write("### Hypotesis 1 and validation")

    st.success(
        f"It is possible to visually differentiate a healthy "
        f"cherry leaf from one with powdery mildew using machine learning."
    )

    st.info(
        f"We will build and train a machine learning model using the dataset "
        f"of cherry leaf images provided by Farmy & Foods. To "
        f"validate this hypothesis, we will evaluate the model's performance "
        f"on a separate test dataset. We expect the model to achieve a "
        f"classification accuracy of at least 97% in distinguishing between "
        f"healthy cherry leaves and those affected by powdery mildew."
        f"This validation will demonstrate the model's ability "
        f"to effectively perform this critical task."
    )

    st.write(
        f"To visually study features of infected and healthy leaves "
        f"visit the **Cherry Leaf Visualiser** page.")

    st.write("### Hypotesis 2 and validation")

    st.success(
        f"Machine learning can predict if a cherry leaf is healthy or "
        f"contains powdery mildew based on leaf images."
    )

    st.info(
        f"We will further validate the model's predictive capabilities by "
        f"assessing its precision, recall, and F1 score on the test dataset."
        f"Achieving an F1 score of at least 0.9 will provide confidence in "
        f"the model's robustness for classifying cherry leaves. This "
        f"validation ensures that the model reliably predicts the presence"
        f" of powdery mildew, a crucial aspect of the project."
    )

    st.write(
        f"To study model performance metrics visit the **ML Performance "
        f"Metrics** page.")

    st.write("### Hypotesis 3 and validation")

    st.success(
        f"A user-friendly dashboard can be developed to provide instant "
        f"cherry leaf health assessments based on uploaded images."
    )

    st.info(
        f"To fulfill this hypothesis, we will design and develop a "
        f"user-friendly web dashboard. During validation, we will conduct"
        f" usability testing with potential end-users to ensure the "
        f"dashboard's intuitiveness and effectiveness. User feedback will "
        f"guide refinements, and the successful deployment of the dashboard, "
        f"integrated with the machine learning model, will demonstrate its "
        f"capability to deliver quick and accurate cherry leaf health "
        f"assessments."
    )

    st.write(
        f"To study the model performance metrics visit the "
        f"**Powdery Mildew Detection** page.")

    st.write("### Hypotesis 4 and validation")

    st.success(
        f"The successful implementation of an ML system for cherry leaf "
        f"assessment can be replicated for other crops, such as pest detection"
        f" across multiple farms."
    )

    st.info(
        f"Once the cherry leaf project proves successful, we will explore "
        f"the potential for replicating the ML-based assessment system on"
        f" other crops within Farmy & Foods' operations. Validation will "
        f"involve piloting the system on different crops and assessing its "
        f"accuracy and scalability. Successful implementation and positive "
        f"feedback from these pilot projects will validate the hypothesis "
        f"and indicate the system's broader applicability for various crops,"
        f" potentially leading to cost savings and improved efficiency "
        f"across multiple farms."
    )
