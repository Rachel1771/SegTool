import streamlit as st

img_path = None

# 创建上传按钮
options = ['Unet', 'Fcn','AttentionUnet']
selected_option = st.selectbox('选择一个模型进行肝脏肿瘤分割', options)