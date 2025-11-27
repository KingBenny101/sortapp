"""Main application entry point for SortApp.

This module configures the Streamlit app and sets up navigation between
the three main pages: Label Images, Settings, and Info.
"""

import streamlit as st

import config
from ui_components import page_label_images, page_settings, page_info

# Configure pages with custom labels
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon="🏷️",
    layout="centered",
    menu_items={
        "About": f"# {config.APP_TITLE}\n\nIncremental learning image classifier with active learning."
    },
)

# Define pages with custom titles
label_page = st.Page(page_label_images, title="Label Images", icon="🏷️")
settings_page = st.Page(page_settings, title="Settings", icon="⚙️")
info_page = st.Page(page_info, title="Info", icon="ℹ️")

# Create navigation
pg = st.navigation([label_page, settings_page, info_page])
pg.run()
