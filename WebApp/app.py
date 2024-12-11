import streamlit as st
from st_pages import add_page_title, get_nav_from_toml
import os



st.set_page_config(layout= "wide")
current_dir = os.getcwd()
st.write("Current working directory:", os.getcwd())


toml_path = os.path.join(current_dir, "WebApp", "pages_sections.toml")
#nav = get_nav_from_toml("streamlitapp/pages_sections.toml")
if not os.path.exists(toml_path):
    st.error("Navigation file not found")
else:
    nav = get_nav_from_toml(toml_path)
    
    if nav:
        pg = st.navigation(nav)
        add_page_title(pg)
        pg.run()
    else:
        st.error("Navigation object is empty, check the format of the toml file")