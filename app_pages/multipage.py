import streamlit as st

bg_img = 'https://i.imgur.com/jZI7V3d.png'
app_icon = (
    'https://img.icons8.com/external-flat-icons-inmotus-design/256/'
    'external-Leaf-ui-flat-icons-inmotus-design.png'
)


def set_app_background(image):
    """
    Defines CSS style for app background adding an image
    """
    style = f"""
    <style>
    .stApp {{
        background-image: url("{image}");
        background-size: 20%;
        background-position: right bottom;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(style, unsafe_allow_html=True)


class MultiPage:
    """
    Class to generate multiple Streamlit pages
    using an object oriented approach
    """

    def __init__(self, app_name) -> None:
        self.pages = []
        self.app_name = app_name

        st.set_page_config(
            page_title=self.app_name,
            page_icon=app_icon)

        set_app_background(bg_img)

    def add_page(self, title, func) -> None:
        self.pages.append({"title": title, "function": func})

    def run(self):
        st.title(self.app_name)
        page = st.sidebar.radio('Menu', self.pages,
                                format_func=lambda page: page['title'])
        page['function']()
