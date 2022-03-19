import streamlit as st

st.markdown("<h1 style='text-align: center; color: red;'>SPAM OR NOT ?</h1>", unsafe_allow_html=True)

# Affichage barre latérale
st.sidebar.markdown("<h1 style='text-align: center; color: red;'>MAIL MENU</h1>", unsafe_allow_html=True)
mail_text = st.sidebar.text_area("Entrez le contenu texte de votre mail :")
option = st.sidebar.selectbox(
     'Quel algorithme choisissez-vous ?',
     ('Email', 'Home phone', 'Mobile phone'))
#img_file_buffer = st.sidebar.camera_input("Ou prenez une photo")
# Fin affichage barre latérale


