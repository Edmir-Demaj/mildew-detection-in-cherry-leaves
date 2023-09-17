import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.image import imread
from src.machine_learning.evaluate_clf import load_test_evaluation


def ml_performance_page():
    version = 'V_1'
    output_dir = f"outputs/{version}/"

    # Displaying Data distribution for train, validation and test sets
    data_distribution = plt.imread(
        f"{output_dir}data_distribution_piechart.png")
    st.image(data_distribution,
             caption="Data distribution for Train, Validation and Test sets")
    st.info(
        f"* The original dataset has a total of 4208 image data files. "
        f"This dataset contains 2104 image files labelled **healthy** "
        f"and 2104 image files labelled **powdery_mildew.**\n\n"
        f"* The dataset was split into Train, Validation and Test set in "
        f"the ratio 0.7, 0.1 and 0.2 respectively."
    )

    st.write('---')

    st.write("### Train, Validation and Test set: Label Frequencies")

    labels_distribution = plt.imread(
        f"outputs/{version}/img_distribution.png")
    st.image(labels_distribution,
             caption='Labels Frequencies for Train, Validation and Test Sets')
    st.info(
        f"From this barplot, we can see that the train, validation and "
        f"test sets are well balanced in terms of class(labels) "
        f"distribution between healthy and powdery_mildew images, as "
        f"they each have an equal number of images for each class. "
        f"This is important to ensure that the model learns to "
        f"distinguish between the two classes equally well and avoids "
        f"bias towards one class. "
    )

    st.write("---")

    st.write("### Model History")
    col1, col2 = st.beta_columns(2)
    with col1:
        # Displaying model training & validation accuracy
        model_acc = plt.imread(f"outputs/{version}/model_training_acc.png")
        st.image(model_acc, caption='Model Training & Validation Accuracy')
    with col2:
        # Displaying model training & validation losses
        model_loss = plt.imread(f"outputs/{version}/model_training_losses.png")
        st.image(model_loss, caption='Model Training & Validation Losses')

    st.info(
        f"A model history plot is an effective visualization tool of the "
        f"performance of an ML model during training and validation. It shows "
        F"the change in model performance over time as the model is trained "
        f"on a given dataset. \n\n The model history plot contains two curves:"
        f" one for the training set (blue color) and one for the validation "
        f"set (orange color). The curve for the "
        f"training set shows how the model's performance improves over"
        f" time as it is exposed to more training data. The curve for the "
        f"validation set shows how the model's performance improves over time "
        f"as it is evaluated on a set of data that it has not seen during "
        f"training.\n\n"
        f"The y-axis of the plot represents the value of the performance "
        f"metric being used(e.g., accuracy, loss, etc.), while the x-axis "
        f"represents the number of epochs(iterations) of training.\n\n"
        f"A good model history plot should show that the model's performance "
        f"on both the training and validation sets is improving over time, "
        f"without overfitting (i.e., the validation accuracy curve should not "
        f"start to degrade after a certain number of epochs or fluctuate "
        f"significantly between epochs, loss curves should not diverge from "
        f"each other, validation accuracy should not plateau"
        f"while the training accuracy continues to increase.)\n\n"
        f"If there is only one significant fluctuation in the loss and "
        f"validation loss curves and no other symptoms of overfitting, it may "
        f"be a sign of a random fluctuation or noise in the data. If the "
        f"model accuracy is consistently high at 99% on the training set and "
        f"the fluctuation in the validation set is relatively small and "
        f"consistent across multiple runs, then it may not be a significant "
        f"cause for concern."
    )

    st.write("---")

    st.write("### Generalised Performance on Test Set")

    # Loading and displaying test set evaluation metrics (loss and accuracy)
    evaluation = load_test_evaluation(version)
    df_evaluation = pd.DataFrame(evaluation, index=['Loss', 'Accuracy'])
    st.dataframe(df_evaluation)

    st.info(
        f"The plot shows the evaluation metrics of the trained ML model on "
        f"the test dataset. The two metrics displayed are loss and accuracy, "
        f"which are important indicators of the model's performance. The loss "
        f"value of 0.0657 indicates how much the predicted values deviate from"
        f" the true values on average, with lower values being better. "
        f"Typically, a loss value of 0.0657 is considered a good value for "
        f"binary classification problems. The accuracy value of 1.0 shows "
        f"the proportion of correctly classified instances in the test "
        f"dataset, with higher values being better.\n\n"
        f"Overall, a low loss value and high accuracy value indicate that the "
        f"machine learning model is performing well on the test dataset and "
        f"has learned to generalize well to new, unseen data. This means that "
        f"the model can be considered robust and reliable, and can be used "
        f"with confidence to make predictions on new data."
    )

    st.write("---")

    # Displaying the confusion matrix
    confusion_matrix = plt.imread(f"outputs/{version}/confusion_matrix.png")
    st.image(confusion_matrix, caption="Confusion Matrix", width=500)

    st.info(
        f"The matrix has four quadrants: true positives (TP), true "
        f"negatives (TN), false positives (FP), and "
        f"false negatives (FN). TP and TN indicate correct predictions, "
        f"while FP and FN indicate incorrect predictions. The accuracy "
        f"metric tells us the percentage of correct predictions made by "
        f"the model - {100*evaluation[1]:.2f}%, while the loss metric - "
        f"{100*evaluation[0]:.2f}% - measures the deviation between the "
        f"predicted and true labels. A high accuracy and low loss "
        f"indicate that the model is making accurate predictions."
    )

    st.write("---")

    st.write("### Classification Report")

    # Displaying the classification report
    classification_report = plt.imread(
        f"outputs/{version}/classification_report.png")
    st.image(classification_report, caption='Classification Report')

    st.info(
        f"The classification report shows that the model has a high "
        f"precision and recall for both classes. **Precision** is the "
        f"number of true positives divided by the sum of true positives "
        f"and false positives. **Recall** is the number of true positives"
        f" divided by the sum of true positives and false negatives. "
        f"High precision indicates that the model can correctly identify "
        f"true positives with a low rate of false positives. High recall "
        f"indicates that the model can correctly identify true positives "
        f"with a low rate of false negatives."
    )

    st.write("---")

    st.write("### ROC Curve")

    # Displaying the ROC curve
    roc_curve = plt.imread(f"outputs/{version}/roc_curve.png")
    st.image(roc_curve, caption='ROC Curve')

    st.info(
        f"The model's ROC curve reveals its impressive ability to accurately "
        f"differentiate between positive and negative samples, with a high "
        f"true positive rate (sensitivity) and low false positive rate (1 - "
        f"specificity) across various thresholds. The AUC score of 1.0 "
        f"confidently demonstrates its exceptional performance in "
        f"distinguishing between the 'Healthy' and 'Powdery Mildew' classes, "
        f"with perfect separation between the two."
    )
