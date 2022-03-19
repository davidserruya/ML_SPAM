import streamlit as st
from PIL import Image


st.markdown("<h1 style='text-align: center; color: red;'>SSL INTERFACE</h1>", unsafe_allow_html=True)
image = Image.open('semisup.png')
st.image(image)
