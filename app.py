import streamlit as st
from rembg import remove, new_session
from PIL import Image
import io
import zipfile
import time

# --- 設定頁面資訊 ---
st.set_page_config(
    page_title="AI 產品圖批次去背神器",
    page_icon="✂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 快取模型 Session ---
# 這樣做可以避免每次去背都重新載入模型，大幅提升速度
# 在雲端環境這尤為重要，能節省記憶體與運算資源
@st.cache_resource
def get_model_session(model_name):
    return new_session(model_name)

# --- 主標題區 ---
st.title("✂️ AI 產品圖批次去背工具")
st.markdown("""
這是一個基於開源 `rembg` (U-2-Net) 技術的自動去背應用。
- **批次處理**：支援一次上傳多張圖片，系統會自動排程處理。
- **一鍵打包**：處理完成後可直接下載 ZIP 壓縮包。
""")

# --- 側邊欄設定 ---
with st.sidebar:
    st.header("⚙️ 設定與上傳")
    
    # 模型選擇 (進階功能，預設 u2net 即可)
    model_name = st.selectbox(
        "選擇去背模型",
        ("u2net", "u2netp", "u2net_human_seg", "isnet-general-use"),
        index=0,
        help="u2net: 預設最強大\nu2netp: 速度快但精度稍低\nu2net_human_seg: 專門針對人像\nisnet-general-use: 通用型，有時邊緣更好"
    )
    
    # 載入模型 Session
    session = get_model_session(model_name)
    
    st.divider()
    
    # 檔案上傳器
    uploaded_files = st.file_uploader(
        "📤 請將圖片拖曳至此 (支援 JPG, PNG, WEBP)", 
        type=['png', 'jpg', 'jpeg', 'webp'], 
        accept_multiple_files=True
    )
    
    st.info(f"💡 提示：建議圖片背景與主體有一定對比度，效果最佳。")

# --- 主邏輯區 ---
if uploaded_files:
    # 顯示處理狀態
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    # 用來儲存結果的列表
    processed_images = []
    
    # 預覽區域 (使用 Expander 收合，避免佔用太多版面)
    with st.expander("👁️ 點擊展開/收合即時預覽 (僅顯示前 10 張)", expanded=True):
        st.write("---")
        
        start_time = time.time()
        
        for i, file in enumerate(uploaded_files):
            # 更新狀態
            status_text.text(f"正在處理第 {i+1} / {len(uploaded_files)} 張圖片: {file.name} ...")
            
            # 1. 讀取圖片
            input_image = Image.open(file)
            
            # 2. 執行去背 (使用快取的 session 加速)
            output_image = remove(input_image, session=session)
            
            # 3. 轉為 Bytes 準備下載
            img_byte_arr = io.BytesIO()
            output_image.save(img_byte_arr, format='PNG')
            img_bytes = img_byte_arr.getvalue()
            
            # 生成新檔名 (原檔名_no_bg.png)
            file_name_no_ext = file.name.rsplit('.', 1)[0]
            new_file_name = f"{file_name_no_ext}_no_bg.png"
            
            processed_images.append((new_file_name, img_bytes))
            
            # 4. 顯示預覽 (限制數量以防瀏覽器卡頓)
            if i < 10:
                col1, col2, col3 = st.columns([1, 1, 0.2])
                with col1:
                    st.image(input_image, caption="原始圖片", use_container_width=True)
                with col2:
                    st.image(output_image, caption="去背結果", use_container_width=True)
                with col3:
                    # 單張下載按鈕
                    st.download_button(
                        label="⬇️",
                        data=img_bytes,
                        file_name=new_file_name,
                        mime="image/png",
                        key=f"btn_{i}"
                    )
                st.divider()
            
            # 更新進度條
            progress_bar.progress((i + 1) / len(uploaded_files))

    end_time = time.time()
    duration = round(end_time - start_time, 2)
    
    # --- 完成後的總結區 ---
    status_text.success(f"✅ 完成！共處理 {len(uploaded_files)} 張圖片，耗時 {duration} 秒。")
    
    # 建立 ZIP 檔
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        for file_name, img_data in processed_images:
            zf.writestr(file_name, img_data)
            
    # 下載全部按鈕 (置中並放大)
    st.markdown("### 📥 下載專區")
    col_dl_1, col_dl_2, col_dl_3 = st.columns([1, 2, 1])
    with col_dl_2:
        st.download_button(
            label=f"📦 下載所有去背圖片 (ZIP 壓縮包) - {len(processed_images)} 張",
            data=zip_buffer.getvalue(),
            file_name="removed_backgrounds.zip",
            mime="application/zip",
            use_container_width=True,
            type="primary"
        )

else:
    # 歡迎畫面
    st.markdown("### 👋 歡迎使用")
    st.markdown("請從左側側邊欄上傳圖片以開始使用。第一次執行時因為需要下載 AI 模型，請耐心等候幾秒鐘。")
    
    # 顯示範例圖 (若有的話，這邊用文字示意)
    st.info("支援批次拖拉上傳，自動打包下載。")