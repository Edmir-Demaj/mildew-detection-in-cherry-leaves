
import streamlit as st
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import random
import glob
import itertools
from matplotlib.image import imread


def leaf_visualizer_page():
    """
    Developes the Cherry leaf visualizer page on Streamlit
    dashboard providing content and funcionality.
    """
    version = 'V_1'
    output_dir = f"outputs/{version}"

    st.write('### Cherry Leaf Visualizer')
    st.info(
        f"The goal here is to visually distinguish a **healthy**"
        f" cherry leaf from one that is infected with **powdery mildew**."
    )

    st.success(
        f"Powdery mildew can cause cherry leaves to develop a white or greyish"
        f" powdery coating. The most noticeable sign is white or greyish marks"
        f" on leaves, often irregular blotches or spots.\n\n"
        f"For effective feature extraction and neural network training with"
        f" image datasets proper image preprocessing is vital. "
        f"This is especially critical when analyzing powdery mildew on leaves."
        f" Normalizing the images in the dataset is crucial before training a "
        f"Neural Network. By calculating the mean and standard deviation of "
        f"the entire dataset and taking into account the visual properties of "
        f"the powdery mildew on the leaf, we can enable the machine "
        f"learning model to accurately and efficiently learn the relevant "
        f"features from the image data."
    )

    if st.checkbox("Difference between average and variability image"):
        avg_healthy = plt.imread(f"{output_dir}/avg_var_healthy.png")
        avg_mildew = plt.imread(f"{output_dir}/avg_var_powdery_mildew.png")

        st.warning(
            f"We noted a distinct variation in color pigmentation between"
            f" the two types of leaves.\n\n"
            f"* Healthy leaves exhibit a consistent and vibrant green color, "
            f" while the leaves affected by powdery mildew display a notable "
            f"deviation in coloration.\n\n"
            f"* Furthermore, the veins in the leaves with powdery mildew are "
            f"more visible compared when contrasted with the healthy leaves."
        )

        st.image(avg_healthy,
                 caption="Healthy Cherry Leaf - Average & Variability")
        st.image(avg_mildew,
                 caption="Powdery Mildew Cherry Leaf - Average & Variability")
        st.write('---')

    if st.checkbox('Differences between Healthy and Powdery Mildew leaves'):
        avg_diff = plt.imread(f"{output_dir}/avg_diff.png")

        st.warning(
            f"* The first two images show the average which is explained "
            f"in the second checkbox."
            f" In the difference between variability, the darker area "
            f"shows where both images are similar while the lighter area "
            f"shows where variability differences. "
        )

        st.image(avg_diff, caption='Difference between average images')
        st.write('---')

    if st.checkbox("Image Montage"):
        st.write(
            "- To view, select a label and click on **Create Montage** button")
        st.write('- To refresh, click on **Create Montage** button')
        data_dir = 'inputs/cherry_leaves_dataset/cherry-leaves/validation'
        labels = os.listdir(data_dir)
        label_to_display = st.selectbox(
            label="Select label",
            options=labels,
            index=0
        )
        if st.button("Create Montage"):
            st.warning(
                f"**Montage Creation Observation**\n\n"
                f"* In the process of generating the montage, consider the "
                f"following: The average dimension of the leaf images is "
                f"approximately 256 pixels in width and 256 pixels "
                f"in height.\n\n"
                f" * Healthy leaves can be distinguished from those affected"
                f" by mildew by the presence of distinctive white streaks,"
                f" adding a contrasting element to your montage."
            )
            image_montage(data_dir, label_to_display,
                          nrows=4, ncols=3, figsize=(10, 12))
            st.write('---')


def image_montage(dir_path, label_to_display, nrows, ncols, figsize=(15, 10)):
    """
    Displays a montage of images from a directory subset based on a given label
    """
    sns.set_style("white")

    # Validate inputs and raise errors respectively
    if not os.path.isdir(dir_path):
        raise ValueError("Invalid directory path.")
    if not isinstance(nrows, int) or not isinstance(ncols, int) \
            or nrows <= 0 or ncols <= 0:
        raise ValueError(
            "Number of rows and columns must be both positive integers.")

    labels = os.listdir(dir_path)
    if label_to_display not in labels:
        raise ValueError(
            f"Label {label_to_display} does not exist in directory.")

    images_list = os.listdir(os.path.join(dir_path, label_to_display))
    if nrows * ncols < len(images_list):
        img_idx = random.sample(images_list, nrows * ncols)
    else:
        print(f"To create a montage, reduce the number of rows or columns. "
              f"Your subset contains a total of {len(images_list)} images, "
              f"but you have requested a montage that includes {nrows * ncols}"
              )
        return

    plot_idx = list(itertools.product(range(nrows), range(ncols)))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    for x, img_file in enumerate(img_idx):
        img_path = os.path.join(dir_path, label_to_display, img_file)
        img = imread(img_path)
        img_shape = img.shape
        axes[plot_idx[x][0], plot_idx[x][1]].imshow(img)
        axes[plot_idx[x][0], plot_idx[x][1]].set_title(
            f"Width {img_shape[1]}px / Height {img_shape[0]}px")
        axes[plot_idx[x][0], plot_idx[x][1]].set_xticks([])
        axes[plot_idx[x][0], plot_idx[x][1]].set_yticks([])

    plt.tight_layout()
    st.pyplot(fig=fig)
