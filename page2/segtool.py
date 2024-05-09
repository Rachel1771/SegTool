import time
import streamlit as st
import os

from st_pages import add_page_title, hide_pages

from page2 import seg_onnx
from PIL import Image
from paddleseg.transforms import Compose, Resize, Normalize

from page2.converpdf import converpdf

# 定义全局空变量
model_path = None
img_path = None
save_dir = None
image_path = None
tier = None
count = 0
area = 0
# st.balloons()
# 按钮切换状态函数
def click_button():
    st.session_state.clicked = True
def change_state():
    st.session_state.clicked = False

add_page_title(layout="wide")

hide_pages(["Thank you"])
st.success("""
感谢以下tools帮助我快速构建了分割工具箱:
* [Paddle](https://www.paddlepaddle.org.cn/)
* [ONNX](https://onnx.ai/)
* [Streamlit](https://streamlit.io/)
""")
st.markdown("🌞本工具箱，使用Paddle平台完成模型训练，使用ONNX进行模型部署，使用Streamlit进行界面设计。")
st.markdown("---")
# 创建一个文件夹用于存储上传的图片
# if not os.path.exists("images"):
#     os.makedirs("images")
# 定义一个全局变量用于存储图片路径

st.markdown("#### 😋下面按钮中上传一张肝脏原图")
# 创建上传按钮
uploaded_file = st.file_uploader("此处上传图片", type=["png"], accept_multiple_files=False, key="upload_file")
if uploaded_file is not None:
    img_path = os.path.join("images", uploaded_file.name)
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())


# options = ['MTDUnet','Unet', 'Fcn','AttentionUnet','Unet++']
options = ['MTDUnet','Unet','Unet++']
st.markdown("#### 🤤在下面按钮中选择一个模型进行分割")
selected_option = st.selectbox('请选择', options)
if selected_option == 'Unet':
    model_path = "model/Unet.onnx"
    save_dir = 'output/Unet'
# elif selected_option == 'Fcn':
#     model_path = "model/Fcn.onnx"
#     save_dir = 'output/Fcn'
# elif selected_option == 'AttentionUnet':
#     model_path = "model/AttentionUnet.onnx"
#     save_dir = 'output/AttentionUnet'
elif selected_option == 'MTDUnet':
    model_path = "model/MTDUnet.onnx"
    save_dir = 'output/PPUnet'
elif selected_option == 'Unet++':
    model_path = "model/Unet++.onnx"
    save_dir = 'output/Unet++'


col1,col2,col3 = st.columns([1,1,1],gap="small")
# 创建图像和分割结果容器
with col1:
    with st.container():
        # st.markdown("<div class='container-wrapper'>", unsafe_allow_html=True)
        image_container = st.container()
        image_container.header("1️⃣原图")
        # st.markdown("</div>", unsafe_allow_html=True)

with col2:
    with st.container():
        # st.markdown("<div class='container-wrapper'>", unsafe_allow_html=True)
        seg_container = st.container()
        seg_container.header("2️⃣着色")
        # st.markdown("</div>", unsafe_allow_html=True)

with col3:
    with st.container():
        pre_container = st.container()
        pre_container.header("3️⃣分割")


if uploaded_file is not None:
    image_container.image(uploaded_file, caption='原图CT', use_column_width=True, output_format='PNG')
    progress_bar = st.progress(0)
    progress_bar.progress(100)
else:
    image_container.image(Image.new('RGB', (512, 512), (255, 255, 255)), caption='没有图片上传', use_column_width=True, output_format='PNG')



if 'clicked' not in st.session_state:
    st.session_state.clicked = False

st.button(' 👉进行分割预测👈', on_click=click_button,use_container_width=True,type="primary")


if st.session_state.clicked:
    if img_path is not None:
        with st.status("拼命加载图片中🚀",expanded=True) as status:
            st.write("加载完毕💪")
            time.sleep(1.5)
            st.write("进行分割中🚀")
            time.sleep(1)
            st.write("结果保存中🚀")
            onnx_model_path = model_path
            image_list = [img_path]
            # save_dir = 'output/Unet'
            transforms = Compose([
                Resize(target_size=(512, 512)),
                Normalize()
            ])
            count,area = seg_onnx.predict(onnx_model_path, transforms, image_list, save_dir=save_dir)
            # st.spinner('Segmenting image...')
            added_saved_dir = os.path.join(save_dir, 'added_prediction')
            pseudo_saved_dir = os.path.join(save_dir, 'pseudo_color_prediction')
            seg = os.path.join(added_saved_dir, uploaded_file.name)
            predict = os.path.join(pseudo_saved_dir, uploaded_file.name)
            seg_image = Image.open(seg)
            pre_image = Image.open(predict).convert('L')
            # image_path = predict
            status.update(label="处理完成啦💪", state="complete", expanded=False)
        seg_container.image(seg_image, caption='肝脏结果', use_column_width=True, output_format='PNG')
        pre_container.image(pre_image, caption='肿瘤结果', use_column_width=True, output_format='PNG')
        st.markdown("---")
        if st.session_state.clicked:
            st.markdown("#### 😜评估结果")
            if (count==0):
                tier = '🟩A+'
                level = 'A+'
            elif (0<count<700):
                tier = '🟨B'
                level = 'B'
            elif (700<count<2700):
                tier = '🟧C'
                level = 'C'
            else:
                tier = '🟥D'
                level = 'D'
            col4,col5 = st.columns(2)
            with col4:
                st.markdown(f"#### 健康等级:{tier}")
            with col5:
                st.markdown(f"#### 预估面积:{round(area,3)}cm^2")
        pdf_file_path = converpdf(img_path,seg,level,area)
        # print(pdf_file_path)
        # st.markdown("#### 😜下载报告")
        with open(pdf_file_path, "rb") as file:
            pdf_name = "报告单.pdf"
            st.download_button(
                label="下载报告",
                data=file,
                file_name=pdf_name,
                mime="application/pdf",
                use_container_width=True,
                # type="primary"
            )
        change_state()
    else:
        seg_container.image(Image.new('RGB', (512, 512), (255, 255, 255)), caption='没有图片上传', use_column_width=True, output_format='PNG')
        pre_container.image(Image.new('RGB', (512, 512), (255, 255, 255)), caption='没有图片上传', use_column_width=True, output_format='PNG')
else:
    seg_container.image(Image.new('RGB', (512, 512), (255, 255, 255)), caption='没有图片上传', use_column_width=True, output_format='PNG')
    pre_container.image(Image.new('RGB', (512, 512), (255, 255, 255)), caption='没有图片上传', use_column_width=True, output_format='PNG')


st.markdown("""
---

## Community notes

* [Notes from Alvaro Navas](https://github.com/ziritrion/dataeng-zoomcamp/blob/main/notes/1_intro.md)
* [Notes from Abd](https://itnadigital.notion.site/Week-1-Introduction-f18de7e69eb4453594175d0b1334b2f4)
* [Notes from Aaron](https://github.com/ABZ-Aaron/DataEngineerZoomCamp/blob/master/week_1_basics_n_setup/README.md)
* [Notes from Faisal](https://github.com/FaisalMohd/data-engineering-zoomcamp/blob/main/week_1_basics_n_setup/Notes/DE%20Zoomcamp%20Week-1.pdf)
* [Michael Harty's Notes](https://github.com/mharty3/data_engineering_zoomcamp_2022/tree/main/week01)
* [Blog post from Isaac Kargar](https://kargarisaac.github.io/blog/data%20engineering/jupyter/2022/01/18/data-engineering-w1.html)
* [Handwritten Notes By Mahmoud Zaher](https://github.com/zaherweb/DataEngineering/blob/master/week%201.pdf)
* [Notes from Candace Williams](https://teacherc.github.io/data-engineering/2023/01/18/zoomcamp1.html)
* [Notes from Marcos Torregrosa](https://www.n4gash.com/2023/data-engineering-zoomcamp-semana-1/)
* [Notes from Vincenzo Galante](https://binchentso.notion.site/Data-Talks-Club-Data-Engineering-Zoomcamp-8699af8e7ff94ec49e6f9bdec8eb69fd)
* [Notes from Victor Padilha](https://github.com/padilha/de-zoomcamp/tree/master/week1)
* [Notes from froukje](https://github.com/froukje/de-zoomcamp/blob/main/week_1_basics_n_setup/notes/notes_week_01.md)
* [Notes from adamiaonr](https://github.com/adamiaonr/data-engineering-zoomcamp/blob/main/week_1_basics_n_setup/2_docker_sql/NOTES.md)
* [Notes from Xia He-Bleinagel](https://xiahe-bleinagel.com/2023/01/week-1-data-engineering-zoomcamp-notes/)
* [Notes from Balaji](https://github.com/Balajirvp/DE-Zoomcamp/blob/main/Week%201/Detailed%20Week%201%20Notes.ipynb)
* [Notes from Erik](https://twitter.com/ehub96/status/1621351266281730049)
* [Notes by Alain Boisvert](https://github.com/boisalai/de-zoomcamp-2023/blob/main/week1.md)
* Notes on [Docker, Docker Compose, and setting up a proper Python environment](https://medium.com/@verazabeida/zoomcamp-2023-week-1-f4f94cb360ae), by Vera
* [Setting up the development environment on Google Virtual Machine](https://itsadityagupta.hashnode.dev/setting-up-the-development-environment-on-google-virtual-machine), blog post by Aditya Gupta 

---

##### 🤪 界面设计来自[2000300309陈乐](https://github.com/Rachel1771)
""")

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True)