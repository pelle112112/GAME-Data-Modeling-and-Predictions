import streamlit as st
import os
from st_pages import add_page_title, get_nav_from_toml


st.set_page_config(layout= "wide")

nav = get_nav_from_toml("../streamlitapp/pages_sections.toml")

pg = st.navigation(nav)


add_page_title(pg)

pg.run()