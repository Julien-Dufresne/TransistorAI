
import streamlit as st

from user_selection_page import (
    render_user_selection,
    render_active_user_sidebar,
)

st.set_page_config(
    page_title="TransistorAI — User Selection Test",
    page_icon="🌍",
)

user = render_user_selection()

if user is not None:
    render_active_user_sidebar()

    st.success(
        f"Welcome, {user['username']}!"
    )

    st.write(f"User ID: `{user['user_id']}`")
