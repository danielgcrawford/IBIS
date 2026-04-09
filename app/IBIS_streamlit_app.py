# app/IBIS_streamlit_app.py
import streamlit as st

st.set_page_config(
    page_title="IBIS",
    layout="wide",
)

st.title("IBIS — Imaging & Analysis")
st.markdown(
    """
Welcome to IBIS.

Use the pages in the left sidebar to view and analyze capture sessions.
"""
)

st.info("Go to **Session Viewer** in the sidebar to load sample sessions.")