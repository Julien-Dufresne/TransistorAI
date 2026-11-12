
"""
TransistorAI — User selection page.

Allows users to create a profile or select an existing
profile before entering the Streamlit application.
"""

import streamlit as st

from src.user_db import (
    create_user,
    get_user,
    get_users,
)


def get_active_user():
    """Return the selected user, or None if no user is active."""
    user_id = st.session_state.get("active_user_id")

    if not user_id:
        return None

    return get_user(user_id)


def render_user_selection():
    """Render the welcome page and return the active user."""

    active_user = get_active_user()

    if active_user is not None:
        return active_user

    st.title("👤 Select Your Profile")
    st.write(
        "Welcome to TransistorInteraction! "
        "Create a new profile or select an existing user "
        "to access the interactive translation and learning tools."
    )

    st.info(
        "Select an existing profile or create a new one "
        "to track your learning progress."
    )

    new_tab, existing_tab = st.tabs([
        "🆕 New User",
        "👤 Existing User",
    ])

    with new_tab:
        st.subheader("Create a profile")

        with st.form("create_user_form"):
            username = st.text_input(
                "Username",
                placeholder="Enter your name",
                max_chars=50,
            )

            submitted = st.form_submit_button(
                "Create profile",
                type="primary",
            )

        if submitted:
            try:
                user = create_user(username)
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.session_state["active_user_id"] = user["user_id"]
                st.rerun()

    with existing_tab:
        st.subheader("Select your profile")

        users = get_users()

        if not users:
            st.info(
                "No profiles exist yet. Create a new user first."
            )
        else:
            user_options = {
                user["username"]: user["user_id"]
                for user in users
            }

            selected_username = st.selectbox(
                "Existing users",
                options=list(user_options.keys()),
                index=None,
                placeholder="Choose a username",
            )

            if st.button(
                "Continue",
                type="primary",
                disabled=selected_username is None,
            ):
                st.session_state["active_user_id"] = (
                    user_options[selected_username]
                )
                st.rerun()

    # No user selected yet.
    return None


def render_active_user_sidebar():
    """Display the active user and allow switching profiles."""

    active_user = get_active_user()

    if active_user is None:
        return

    with st.sidebar:
        st.divider()
        st.write(f"👤 **{active_user['username']}**")

        if st.button(
            "Switch User",
            use_container_width=True,
        ):
            # Clear the selected identity.
            st.session_state.pop("active_user_id", None)

            for key in [
                "translation_results",
                "translation_errors",
                "translation_input",
                "translation_source",
            ]:
                st.session_state.pop(key, None)

            # Clear widget state when switching profiles.
            for key in list(st.session_state.keys()):
                if key.startswith("answer_"):
                    del st.session_state[key]

            st.rerun()
