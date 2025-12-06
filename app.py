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
    
    # 定義模型詳細說明資料
    model_descriptions = {
        "u2net": {
            "label": "u2net (標準通用版)",
            "details": """
            **特點**：這是 U-2-Net 的原始標準模型。  
            **優點**：泛用性最高，對大多數物體（人、動物、商品、車輛）都有不錯的效果。  
            **缺點**：模型檔案較大（約 170MB），運算速度比輕量版稍慢。  
            **適用情境**：大多數情況的首選。如果你不確定要選哪個，先用這個。
            """
        },
        "u2netp": {
            "label": "u2netp (輕量快速版)",
            "details": """
            **特點**：P 代表 Portable（便攜/輕量化），是 u2net 的縮小版。  
            **優點**：檔案非常小（約 4MB），運算速度非常快，幾乎不佔記憶體。  
            **缺點**：精細度較差，對於邊緣複雜的物體（如髮絲、網狀物）去背效果不如標準版，邊緣可能會比較生硬。  
            **適用情境**：手機端應用、低階電腦，或者你需要批次處理幾千張圖片且對邊緣要求不高時。
            """
        },
        "u2net_human_seg": {
            "label": "u2net_human_seg (人像專用版)",
            "details": """
            **特點**：專門針對「人類」進行訓練的模型。  
            **優點**：在處理人物照片時表現最好，對於頭髮、衣服皺褶的判斷比通用版準確。  
            **缺點**：對非人類物體（如桌子、汽車、貓狗）的效果可能很差。  
            **適用情境**：只用來處理人像（如證件照、模特兒照片）。
            """
        },
        "isnet-general-use": {
            "label": "isnet-general-use (高細節通用版)",
            "details": """
            **特點**：這是基於較新的 IS-Net 架構，通常被視為 u2net 的升級替代品。  
            **優點**：對於「細微邊緣」（如飄逸的髮絲、動物毛髮、半透明物體）的處理能力通常比 u2net 更好，邊緣過渡更自然。  
            **適用情境**：高品質去背推薦用這個。特別是當你要去背的物體有複雜邊緣（毛茸茸的玩偶、頭髮很多的人、植物）時。
            """
        }
    }

    # 模型選擇選單 (使用 label 作為顯示名稱)
    selected_model_key = st.selectbox(
        "選擇去背模型",
        options=list(model_descriptions.keys()),
        format_func=lambda x: model_descriptions[x]["label"],
        index=0
    )

    # 動態顯示選定模型的詳細說明
    st.info(model_descriptions[selected_model_key]["details"])

    # 快速選擇指南 (懶人包) - 使用 Expander 收合
    with st.expander("📖 快速選擇指南 (懶人包)"):
        st.markdown("""
        | 你的需求 | 推薦選擇 |
        | :--- | :--- |
        | 不知道選哪個 / 什麼都去 | **u2net** 或 **isnet-general-use** |
        | 追求最高畫質 / 有毛髮細節 | **isnet-general-use** (大推 👍) |
        | 只處理人像 / 模特兒 | **u2net_human_seg** |
        | 電腦跑不動 / 需要極速處理 | **u2netp** |
        """)
    
    # 載入模型 Session
    session = get_model_session(selected_model_key)
    
    st.divider()
    
    # 檔案上傳器
    uploaded_files = st.file_uploader(
        "📤 請將圖片拖曳至此 (支援 JPG, PNG, WEBP)", 
        type=['png', 'jpg', 'jpeg', 'webp'], 
        accept_multiple_files=True
    )
    
    st.caption(f"💡 提示：建議圖片背景與主體有一定對比度，效果最佳。")

# --- 主邏輯區 ---
if uploaded_files:
    # 顯示處理狀態
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    # 用來儲存結果的列表
    processed_images = []
    
    # 預覽區域
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
            
    # 下載全部按鈕
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
    st.markdown("請從左側側邊欄上傳圖片以開始使用。第一次使用特定模型時，系統會自動下載模型檔案，請稍候。")
    
    st.info("支援批次拖拉上傳，自動打包下載。")
