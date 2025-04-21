import streamlit as st
import Home, app


def main():
    st.sidebar.title("Menu")
    PAGES = {
        "🏠 Home": Home,
        "🔍 PredictAuto": app,
    }


    page_name = st.sidebar.radio("", list(PAGES.keys()))
    page = PAGES[page_name]
    with st.spinner(f"Chargement de {page_name} ..."):
        page.upload_predict_page()


if __name__ == "__main__":
    main()