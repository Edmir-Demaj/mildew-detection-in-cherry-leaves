import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from src.data_management import download_df_as_csv
from src.machine_learning.predictive_analysis import (
    resize_input_image,
    make_prediction,
    plot_prediction_probabilities
)


def powdery_mildew_detection_page():
    """
    Developes the Mildew detection page on Streamlit
    dashboard providing content and funcionality.
    """
    st.info(
        f"In this platform, you have the opportunity to "
        f"submit high-resolution images of Cherry leaves for "
        f"precise identification of powdery mildew infection or fungus. Once "
        f"your images are processed, you can easily obtain a comprehensive "
        f"report detailing the outcomes of the analysis."
    )

    st.write(
        f"You can download the Cherry leaves dataset for live prediction "
        f"[here](https://www.kaggle.com/datasets/codeinstitute/cherry-leaves)."
    )

    st.write("---")

    images_buffer = st.file_uploader(f'Upload Cherry leaves images. You may '
                                     f'select more than one image.',
                                     type=['png', 'jpg',
                                           'jpeg', 'bmp', 'webp'],
                                     accept_multiple_files=True)

    # Upload image files, resize it and run ml model to
    # make prdiction for image data uploaded.
    if images_buffer is not None:
        df_report = pd.DataFrame([])
        for image in images_buffer:

            img_pil = (Image.open(image)).convert('RGB')
            st.info(f"Cherry Leaf Image: **{image.name}**")
            img_array = np.array(img_pil)
            st.image(img_pil,
                     caption=f"Image Size: {img_array.shape[1]}px width "
                             f"/ {img_array.shape[0]}px height")

            version = 'V_1'
            resized_img = resize_input_image(img=img_pil, version=version)
            pred_proba, pred_class = make_prediction(resized_img,
                                                     version=version)
            plot_prediction_probabilities(pred_proba, pred_class)

            df_report = df_report.append({"Name": image.name,
                                          'Result': pred_class},
                                         ignore_index=True)
                                         
        # Display analysis report and download option
        if not df_report.empty:
            st.success("Analysis Results Report")
            st.table(df_report)
            st.markdown(download_df_as_csv(df_report),
                        unsafe_allow_html=True)
