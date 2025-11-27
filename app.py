import streamlit as st

import config
from ui_components import page_label_images, page_info


# ------------------------------
# MAIN
# ------------------------------
def main():
    st.set_page_config(page_title=config.APP_TITLE, layout="wide")

    with st.sidebar:
        page = st.radio(
            "Page",
            ["Label images", "Info & paths"],
            index=0,
        )

    if page == "Label images":
        page_label_images()
    else:
        page_info()


if __name__ == "__main__":
    main()
