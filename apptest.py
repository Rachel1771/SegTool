import time
import streamlit as st
import os
from page2 import seg_onnx
from PIL import Image
from paddleseg.transforms import Compose, Resize, Normalize

# 创建一个文件夹用于存储上传的图片
# if not os.path.exists("images"):
#     os.makedirs("images")

# 设置页面标题
st.set_page_config(page_title="Liver Segmentation")


# 定义一个全局变量用于存储图片路径
img_path = None

# 创建上传按钮
uploaded_file = st.file_uploader("请上传一张肝脏图片", type=["png"], accept_multiple_files=False, key="upload_file")
if uploaded_file is not None:
    # img_path = os.path.join("images", uploaded_file.name)
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

model_path = None
save_dir = None
options = ['Unet', 'Fcn','AttentionUnet']
selected_option = st.selectbox('选择一个模型进行肝脏肿瘤分割', options)
if selected_option == 'Unet':
    model_path = "model/Unet.onnx"
    save_dir = 'output/Unet'
elif selected_option == 'Fcn':
    model_path = "model/Fcn.onnx"
    save_dir = 'output/Fcn'
elif selected_option == 'AttentionUnet':
    model_path = "model/AttentionUnet.onnx"
    save_dir = 'output/AttentionUnet'

col1,col2,col3 = st.columns([1,1,1],gap="small")
# 创建图像和分割结果容器
with col1:
    with st.container():
        # st.markdown("<div class='container-wrapper'>", unsafe_allow_html=True)
        image_container = st.container()
        image_container.header("原图")
        # st.markdown("</div>", unsafe_allow_html=True)

with col2:
    with st.container():
        # st.markdown("<div class='container-wrapper'>", unsafe_allow_html=True)
        seg_container = st.container()
        seg_container.header("肝脏")
        # st.markdown("</div>", unsafe_allow_html=True)

with col3:
    with st.container():
        pre_container = st.container()
        pre_container.header("肿瘤")
# 相框1 - 显示上传的图片


if uploaded_file is not None:
    image_container.image(uploaded_file, caption='原图CT', use_column_width=True, output_format='PNG')
    progress_bar = st.progress(0)
    progress_bar.progress(100)
else:
    image_container.image(Image.new('RGB', (512, 512), (255, 255, 255)), caption='没有图片上传', use_column_width=True, output_format='PNG')



if 'clicked' not in st.session_state:
    st.session_state.clicked = False
def click_button():
    st.session_state.clicked = True

st.button('👉进行分割预测👈', on_click=click_button,use_container_width=True,type="primary")
# 相框2 - 显示预测分割结果
# if img_path is not None:
if st.session_state.clicked:
    if img_path is not None:
        with st.status("加载图片中...",expanded=True) as status:
            st.write("加载完毕")
            time.sleep(1.5)
            st.write("进行分割")
            time.sleep(1)
            st.write("结果保存中...")
            onnx_model_path = model_path
            image_list = [img_path]
            # save_dir = 'output/Unet'
            transforms = Compose([
                Resize(target_size=(512, 512)),
                Normalize()
            ])
            seg_onnx.predict(onnx_model_path, transforms, image_list, save_dir=save_dir)
            # st.spinner('Segmenting image...')
            added_saved_dir = os.path.join(save_dir, 'added_prediction')
            pseudo_saved_dir = os.path.join(save_dir, 'pseudo_color_prediction')
            seg = os.path.join(added_saved_dir, uploaded_file.name)
            predict = os.path.join(pseudo_saved_dir, uploaded_file.name)
            seg_image = Image.open(seg)
            pre_image = Image.open(predict)
            status.update(label="处理完成啦！", state="complete", expanded=False)
        seg_container.image(seg_image, caption='肝脏结果', use_column_width=True, output_format='PNG')
        pre_container.image(pre_image, caption='肿瘤结果', use_column_width=True, output_format='PNG')
    else:
        seg_container.image(Image.new('RGB', (512, 512), (255, 255, 255)), caption='没有图片上传', use_column_width=True, output_format='PNG')
        pre_container.image(Image.new('RGB', (512, 512), (255, 255, 255)), caption='没有图片上传', use_column_width=True, output_format='PNG')
else:
    seg_container.image(Image.new('RGB', (512, 512), (255, 255, 255)), caption='没有图片上传', use_column_width=True, output_format='PNG')
    pre_container.image(Image.new('RGB', (512, 512), (255, 255, 255)), caption='没有图片上传', use_column_width=True, output_format='PNG')

# 页面底部信息
st.markdown("<hr>", unsafe_allow_html=True)
st.write("<p style='text-align: center; color: #777;'>This application was developed by [Your Name] as part of a graduate design project.</p>", unsafe_allow_html=True)

css_style = """
button[type=submit] {
    background-color: #1e40af;
    color: #dbeafe;
    padding: 12px 320px;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    margin-top: 1.5rem;
}

button[type=submit]:hover {
    background-color: #1d4ed8;
}
"""
st.write(f"<style>{css_style}</style>", unsafe_allow_html=True)