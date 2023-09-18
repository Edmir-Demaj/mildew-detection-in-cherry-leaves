# Powdery Mildew Detection in Cherry Leaves (ML project)

![mockup](assets/readme_images/mockup.png)

Live application is available [here.](https://ml-mildew-detection-8675ca112542.herokuapp.com/)

This project employs data science and machine learning to detect powdery mildew on cherry leaves, distinguishing between healthy and diseased ones. It features a binary classification machine learning model accessible via a Streamlit dashboard for leaf health prediction. Additionally, the project includes traditional data analysis findings, detailed hypothesis examination, and model performance evaluation.

To streamline the workflow, the project utilizes three Jupyter notebooks: one for data import and cleaning, another for data visualization, and the third for developing and assessing a TensorFlow deep learning model. These notebooks provide a structured approach to data management and model development.

The primary goal is to improve powdery mildew detection on cherry leaves, aiding growers in effectively identifying and treating diseased cherry trees. Leveraging machine learning and data analysis, this project aims to offer an efficient, user-friendly solution for distinguishing between healthy and diseased cherry leaves.
___

## Table of Contents

- [Powdery Mildew Detection in Cherry Leaves (ML project)](#powdery-mildew-detection-in-cherry-leaves-ml-project)
  - [Table of Contents](#table-of-contents)
- [Planning Phase](#planning-phase)
  - [Agile methodology - Development](#agile-methodology---development)
  - [Crisp-DM: Definition and Usage](#crisp-dm-definition-and-usage)
  - [Business Requirements](#business-requirements)
  - [**Hypothesis and Validation**](#hypothesis-and-validation)
- [Data Gathering Phase](#data-gathering-phase)
  - [Dataset Content](#dataset-content)
  - [Rationale to map the business requirements to the Data Visualizations and ML tasks](#rationale-to-map-the-business-requirements-to-the-data-visualizations-and-ml-tasks)
  - [Data Understanding](#data-understanding)
- [Project Execution Phase](#project-execution-phase)
  - [Data Preparation](#data-preparation)
---

# Planning Phase

## Agile methodology - Development

- Agile with Kanban is chosen for project management to promote flexibility, adaptability, and efficient issue tracking throughout the project's lifecycle.
- Find the Kanban board Project [here.](https://github.com/users/Edmir-Demaj/projects/9)

![kanban](assets/readme_images/kanban_project.png)


## Crisp-DM: Definition and Usage

CRISP-DM (Cross-Industry Standard Process for Data Mining) is a structured approach to data mining and analytics projects, also known as a framework. It consists of the following stages:

1. **Understand:** Define the business problem, objectives, and goals.
2. **Data:** Collect and explore data, ensuring its quality and relevance.
3. **Prepare:** Clean, transform, and engineer data for analysis.
4. **Model:** Select and build appropriate predictive models.
5. **Evaluate:** Assess model performance and alignment with project goals.
6. **Deploy:** Implement the chosen model into production.
7. **Monitor:** Continuously track and maintain model performance.

![crips-dm](assets/readme_images/crisp_dm.png)


## Business Requirements

The cherry plantation crop from Farmy & Foods faces a challenge where their cherry plantations have been presenting powdery mildew. Currently, the process is to verify if a given cherry tree contains powdery mildew manually. An employee spends around 30 minutes in each tree, taking a few samples of tree leaves and demonstrating visually if the leaf tree is healthy or has powdery mildew. If it has powdery mildew, the employee applies a specific compound to kill the fungus. The time spent using this compound is 1 minute.  The company has thousands of cherry trees on multiple farms nationwide. As a result, this manual process could be more scalable due to the time spent in the manual process inspection.

To save time, the IT team suggested an ML system that can detect instantly, using a leaf tree image, if it is healthy or has powdery mildew. A similar manual process is in place for other crops for detecting pests. If this initiative is successful, there is a realistic chance to replicate this project in all other crops. The dataset is a collection of cherry leaf images provided by Farmy & Foods, taken from their crops.

- 1 - The client is interested in conducting a study to visually differentiate a healthy cherry leaf from one with powdery mildew.
- 2 - The client is interested in predicting if a cherry leaf is healthy or contains powdery mildew.
- 3 - The client wants a dashboard that meets the above requirements.

## **Hypothesis and Validation**

- **Hypothesis one**
  - It is possible to visually differentiate a healthy cherry leaf from one with powdery mildew using machine learning.
- **Validation:** <br>
    We will build and train a machine learning model using the dataset of cherry leaf images provided by Farmy & Foods. To validate this hypothesis, we will evaluate the model's performance on a separate test dataset. We expect the model to achieve a classification accuracy of at least 97% in distinguishing between healthy cherry leaves and those affected by powdery mildew. This validation will demonstrate the model's ability to effectively perform this critical task.

- **Hypothesis Two**
  - Machine learning can predict if a cherry leaf is healthy or contains powdery mildew based on leaf images.
- **Validation:** <br>
    We will further validate the model's predictive capabilities by assessing its precision, recall, and F1 score on the test dataset. Achieving an F1 score of at least 0.9 will provide confidence in the model's robustness for classifying cherry leaves. This validation ensures that the model reliably predicts the presence of powdery mildew, a crucial aspect of the project.
- **Hypothesis Three**
  - A user-friendly dashboard can be developed to provide instant cherry leaf health assessments based on uploaded images.
- **Validation:** <br>
    To fulfill this hypothesis, we will design and develop a user-friendly web dashboard. During validation, we will conduct usability testing with potential end-users to ensure the dashboard's intuitiveness and effectiveness. User feedback will guide refinements, and the successful deployment of the dashboard, integrated with the machine learning model, will demonstrate its capability to deliver quick and accurate cherry leaf health assessments.
- **Hypothesis Four**
  - The successful implementation of an ML system for cherry leaf assessment can be replicated for other crops, such as pest detection, across multiple farms.
- **Validation:** <br>
    Once the cherry leaf project proves successful, we will explore the potential for replicating the ML-based assessment system on other crops within Farmy & Foods' operations. Validation will involve piloting the system on different crops and assessing its accuracy and scalability. Successful implementation and positive feedback from these pilot projects will validate the hypothesis and indicate the system's broader applicability for various crops, potentially leading to cost savings and improved efficiency across multiple farms.
    
    ---

# Data Gathering Phase

## Dataset Content
    
* The dataset is sourced from Kaggle. We then created a fictitious user story where predictive analytics can be applied in a real project in the workplace.
* The dataset contains +4 thousand images taken from the client's crop fields. The images show healthy cherry leaves and cherry leaves that have powdery mildew, a fungal disease that affects many plant species. The cherry plantation crop is one of the finest products in their portfolio, and the company is concerned about supplying the market with a compromised quality product.

## Rationale to map the business requirements to the Data Visualizations and ML tasks

*Business Requirement 1:*

As a client, I can navigate easily around an interactive dashboard so that I can view and understand the data presented.
As a client, I can view an image montage of either healthy or powdery mildew-affected cherry leaves so that I can visually differentiate them.
As a client, I can view and toggle visual graphs of average images (and average image difference) and image variabilities for both healthy and powdery mildew-affected cherry leaves so that I can observe the difference and understand the visual markers that indicate leaf quality better.

*Business Requirement 2:*

As a client, I can access and use a machine learning model so that I can obtain a class prediction on a cherry leaf image provided.
As a client, I can provide new raw data on a cherry leaf and clean it so that I can run the provided model on it.
As a client, I can feed cleaned data to the dashboard to allow the model to predict it so that I can instantly discover whether a given cherry leaf is healthy or affected by powdery mildew.
As a client, I can save model predictions in a timestamped CSV file so that I can keep an account of the predictions that have been made.

*Business Requirement 3:*

As a client, I can view an explanation of the project's hypotheses so that I can understand the assumptions behind the machine learning model and its predictions.
As a client, I can view a performance evaluation of the machine learning model so that I can assess its accuracy and effectiveness.
As a client, I can access pages containing the findings from the project's conventional data analysis so that I can gain additional insights into the data and its patterns.

## Data Understanding

The dataset consists of labeled image data divided into two folders, with each folder representing a specific image label. Specifically, images of healthy leaves are located in the "healthy" directory, while images of leaves with powdery mildew are stored in the "powder_mildew" directory.

In total, the classification dataset comprises 4,208 records, evenly split between 2,104 images of healthy leaves and 2,104 images of infected leaves. This balanced dataset ensures equal representation of both classes for effective model training and evaluation.

---

# Project Execution Phase

## Data Preparation
Minimal data cleaning was required, and the folders were scanned through to delete any non-image files. The dataset was split into the train, test and validation sets to perform model training and avoid model overfitting adequately. The split ratio of the dataset was 0.7, 0.2, and 0.1, respectively.
Data augmentation was performed using ImageDataGenerator on the training dataset to increase the image data by artificially and temporarily creating training images through the combination of different processes, such as random rotation, shifts, sheared, zoom and rotated images in the computer's short-term memory (RAM). ImageDataGenertor was also used to rescale the test dataset and validation dataset.