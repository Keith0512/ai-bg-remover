# Version: v2.4 (Robust Clipboard & JSON Fix)
import streamlit as st
from rembg import remove, new_session
from PIL import Image
import io
import time
import json
import base64
import gc
import re
import streamlit.components.v1 as components

# --- 引入 Google 官方 SDK ---
from google import genai
from google.genai import types

# --- 設定頁面資訊 ---
st.set_page_config(
    page_title="AI 電商圖一條龍生成器",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 常數設定 ---
PRO_TEXT_MODEL = "gemini-3-pro-preview"
PRO_IMAGE_MODEL = "gemini-3-pro-image-preview"
FLASH_TEXT_MODEL = "gemini-2.5-flash-preview-09-2025"
FLASH_IMAGE_MODEL = "gemini-2.5-flash-image-preview"

# --- JS 元件：複製圖片到剪貼簿 (權限增強版) ---
def copy_image_button(image_bytes, key_suffix):
    b64_str = base64.b64encode(image_bytes).decode()
    
    # 這裡的 HTML/JS 會在 iframe 中執行
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ margin: 0; padding: 0; display: flex; justify-content: center; align-items: center; height: 100%; }}
            .copy-btn {{
                background-color: #f0f2f6; 
                border: 1px solid #d0d0d0; 
                border-radius: 4px; 
                padding: 5px 10px; 
                cursor: pointer; 
                font-size: 14px; 
                font-family: sans-serif;
                display: flex; 
                align-items: center; 
                gap: 5px;
                color: #31333F;
                text-decoration: none;
                transition: background-color 0.2s;
            }}
            .copy-btn:hover {{ background-color: #e0e0e0; }}
            .copy-btn:active {{ background-color: #d0d0d0; }}
            .msg {{ margin-left: 8px; font-size: 12px; font-family: sans-serif; }}
        </style>
    </head>
    <body>
        <button id="btn" class="copy-btn" onclick="copyImage()">
            📋 複製圖片
        </button>
        <span id="msg" class="msg"></span>

        <script>
        async function copyImage() {{
            const btn = document.getElementById("btn");
            const msg = document.getElementById("msg");
            
            msg.innerText = "⏳...";
            msg.style.color = "gray";

            try {{
                // 1. 檢查 Clipboard API 支援度
                if (!navigator.clipboard || !navigator.clipboard.write) {{
                    throw new Error("API_NOT_SUPPORTED");
                }}

                // 2. 將 Base64 轉為 Blob
                const response = await fetch("data:image/png;base64,{b64_str}");
                const blob = await response.blob();
                
                // 3. 寫入剪貼簿
                const item = new ClipboardItem({{ "image/png": blob }});
                await navigator.clipboard.write([item]);
                
                msg.innerText = "✅ 已複製！";
                msg.style.color = "green";
                
            }} catch (err) {{
                console.error("Copy failed:", err);
                if (err.message === "API_NOT_SUPPORTED") {{
                    msg.innerText = "❌ 瀏覽器不支援";
                }} else {{
                    msg.innerText = "❌ 失敗 (請手動下載)";
                }}
                msg.style.color = "red";
            }} finally {{
                setTimeout(() => {{ 
                    if(msg.innerText.includes("已複製")) msg.innerText = "";
                }}, 2500);
            }}
        }}
        </script>
    </body>
    </html>
    """
    # height 設定為 45px 剛好容納按鈕
    components.html(html_code, height=45)

# --- 記憶體優化輔助函式 ---
def pil_to_bytes(image, format="PNG", quality=95):
    buf = io.BytesIO()
    if format == "JPEG":
        if image.mode in ('RGBA', 'LA'):
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1])
            image = background
        image.save(buf, format=format, quality=quality)
    else:
        image.save(buf, format=format)
    return buf.getvalue()

def bytes_to_pil(image_bytes):
    return Image.open(io.BytesIO(image_bytes))

# --- 高品質放大函式 (Upscaling) ---
def upscale_image(image, scale_factor=2):
    """使用 Lanczos 演算法進行高品質放大"""
    new_size = (int(image.width * scale_factor), int(image.height * scale_factor))
    return image.resize(new_size, Image.Resampling.LANCZOS)

# --- 圖片縮小保護 (SDK 雖然方便，但為了省錢還是要縮) ---
def resize_image_for_api(image, max_size=(1024, 1024)):
    img_copy = image.copy()
    img_copy.thumbnail(max_size, Image.Resampling.LANCZOS)
    return img_copy

# --- API Key 淨化 ---
def clean_api_key(key):
    if not key: return ""
    return re.sub(r'[^a-zA-Z0-9\-\_]', '', key.strip())

# --- 核心功能：驗證 API Key (使用 SDK) ---
def check_pro_model_access(api_key):
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=PRO_TEXT_MODEL,
            contents="Ping",
            config=types.GenerateContentConfig(max_output_tokens=1)
        )
        return response is not None
    except Exception as e:
        return False

# --- 分析函式 (使用 SDK + 防呆機制) ---
def analyze_image_with_gemini(api_key, image, model_name):
    processed_img = resize_image_for_api(image)
    
    prompt = """
    你是一位專業的電商視覺總監。
    請分析這張已經去背的商品圖片，並構思 5 個能大幅提升轉化率的「高階商品攝影場景」。
    請回傳一個純 JSON Array (不要 Markdown)，格式如下：
    [ { "title": "風格標題", "prompt": "詳細的英文生圖提示詞...", "reason": "使用繁體中文解釋為什麼適合此商品" }, ... ]
    
    設計方向：
    1. 極簡高奢 (Minimalist High-End)
    2. 真實生活感 (Authentic Lifestyle)
    3. 幾何藝術 (Abstract Geometric)
    4. 自然有機 (Nature & Organic)
    5. AI 獨家推薦 (AI Recommendation - 根據商品特性，自由發揮一個最獨特且賣座的場景，標題開頭請加 '🤖 AI推薦：')
    
    【重要指令】：
    1. 所有的 prompt 結尾必須強制包含以下高品質關鍵詞：
    "High resolution, 8k, extreme detail, product photography masterpiece, sharp focus, professional lighting, cinematic composition"
    2. "reason" 欄位必須使用 **繁體中文** 撰寫。
    """
    
    client = genai.Client(api_key=api_key)
    
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=[prompt, processed_img],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        return json.loads(response.text)
        
    except Exception as e:
        if model_name == PRO_TEXT_MODEL:
            st.toast(f"⚠️ Pro 模型異常 ({str(e)})，自動降級...", icon="🔄")
            try:
                response = client.models.generate_content(
                    model=FLASH_TEXT_MODEL,
                    contents=[prompt, processed_img],
                    config=types.GenerateContentConfig(response_mime_type="application/json")
                )
                return json.loads(response.text)
            except Exception as e2:
                raise Exception(f"分析失敗 (Flash 也失敗): {str(e2)}")
        else:
            raise Exception(f"分析失敗: {str(e)}")

# --- 生成函式 (使用 SDK) ---
def generate_image_with_gemini(api_key, product_image, base_prompt, model_name, user_extra_prompt="", ref_image=None):
    processed_product = resize_image_for_api(product_image)
    
    full_prompt = f"""
    Professional product photography masterpiece.
    Subject: The FIRST image provided is the PRODUCT. KEEP THE PRODUCT APPEARANCE EXACTLY AS IS.
    """
    if ref_image:
        full_prompt += "\nReference: The SECOND image provided is a STYLE/CHARACTER REFERENCE. Integrate the product into a scene consistent with this reference."
    
    full_prompt += f"\nBackground & Atmosphere: {base_prompt}"
    if user_extra_prompt:
        full_prompt += f"\nAdditional User Requirements: {user_extra_prompt}"
    
    full_prompt += "\nQuality: 8k ultra-high resolution, extreme detail, 4000px, sharp focus, macro details, commercial standard, ray tracing."

    contents = [full_prompt, processed_product]
    if ref_image:
        contents.append(resize_image_for_api(ref_image))

    client = genai.Client(api_key=api_key)
    
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=types.GenerateContentConfig(response_modalities=["IMAGE"])
        )
        
        for part in response.candidates[0].content.parts:
            if part.inline_data:
                return Image.open(io.BytesIO(part.inline_data.data))
        
        raise Exception("模型未回傳圖片數據")

    except Exception as e:
        if model_name == PRO_IMAGE_MODEL:
            st.toast(f"⚠️ Pro 模型異常，自動切換至 Flash...", icon="🔄")
            try:
                response = client.models.generate_content(
                    model=FLASH_IMAGE_MODEL,
                    contents=contents,
                    config=types.GenerateContentConfig(response_modalities=["IMAGE"])
                )
                for part in response.candidates[0].content.parts:
                    if part.inline_data:
                        return Image.open(io.BytesIO(part.inline_data.data))
                raise Exception("Flash 模型也未回傳圖片")
            except Exception as e2:
                raise Exception(f"生成失敗 (雙重失敗): {str(e2)}")
        else:
            raise Exception(f"生成失敗: {str(e)}")

# --- Session 初始化 ---
@st.cache_resource
def get_model_session(name): return new_session(name)

if 'processed_images' not in st.session_state: st.session_state.processed_images = {}
if 'prompts' not in st.session_state: st.session_state.prompts = {}
if 'generated_results' not in st.session_state: st.session_state.generated_results = {}
if 'last_validated_key' not in st.session_state: st.session_state.last_validated_key = None
if 'user_model_tier' not in st.session_state: st.session_state.user_model_tier = "FLASH"

# --- 側邊欄 ---
with st.sidebar:
    st.header("⚙️ 設定")
    raw_api_key = st.text_input("Google API Key (選填)", type="password")
    user_api_key = clean_api_key(raw_api_key)
    
    final_api_key = user_api_key if user_api_key else st.secrets.get("GEMINI_API_KEY", "")
    final_api_key = clean_api_key(final_api_key)
    
    if user_api_key and user_api_key != st.session_state.last_validated_key:
        with st.spinner("驗證 Pro 權限..."):
            if check_pro_model_access(user_api_key):
                st.session_state.user_model_tier = "PRO"
                st.toast("✅ Pro 權限已啟用", icon="🚀")
            else:
                st.session_state.user_model_tier = "FLASH"
                st.error("⚠️ 無法啟用 Pro (未綁定帳單)，降級為 Flash")
            st.session_state.last_validated_key = user_api_key
    elif not user_api_key:
        st.session_state.user_model_tier = "FLASH"
        st.session_state.last_validated_key = None

    current_text_model = PRO_TEXT_MODEL if st.session_state.user_model_tier == "PRO" and user_api_key else FLASH_TEXT_MODEL
    
    if st.session_state.user_model_tier == "PRO" and user_api_key:
        st.success(f"🚀 **Pro Mode** (Vision: {PRO_TEXT_MODEL})")
    else:
        st.info(f"⚡ **Flash Mode** (Vision: {FLASH_TEXT_MODEL})")

    st.divider()
    model_labels = {"u2netp": "u2netp (快速-推薦)", "isnet-general-use": "isnet (高細節)", "u2net": "u2net (標準)"}
    sel_mod = st.selectbox("去背模型", list(model_labels.keys()), format_func=lambda x: model_labels[x], index=0)
    session = get_model_session(sel_mod)
    st.divider()
    st.caption("v2.4 (Robust Clipboard & JSON Fix)")

# --- 主畫面 ---
uploaded_files = st.file_uploader("1️⃣ 上傳商品圖片", type=['png', 'jpg', 'jpeg', 'webp'], accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        if file.name not in st.session_state.processed_images:
            with st.spinner(f"正在去背: {file.name}..."):
                img = Image.open(file)
                if max(img.size) > 1024: img.thumbnail((1024, 1024))
                out = remove(img, session=session)
                st.session_state.processed_images[file.name] = {
                    "original_data": pil_to_bytes(img, "JPEG"),
                    "nobg_data": pil_to_bytes(out, "PNG")
                }
                del img, out
                gc.collect()

    st.divider()
    if st.session_state.processed_images:
        st.subheader("2️⃣ AI 分析與生成")
        selected_file_name = st.selectbox("選擇商品", list(st.session_state.processed_images.keys()))
        
        if selected_file_name:
            curr = st.session_state.processed_images[selected_file_name]
            nobg_pil = bytes_to_pil(curr["nobg_data"])
            
            c1, c2 = st.columns(2)
            with c1: st.image(bytes_to_pil(curr["original_data"]), caption="原始", use_container_width=True)
            with c2: st.image(nobg_pil, caption="去背", use_container_width=True)
            
            d1, d2 = st.columns([1, 1])
            with d1: st.download_button("⬇️ 下載去背圖", curr["nobg_data"], f"{selected_file_name}_nobg.png", "image/png", use_container_width=True)
            with d2: copy_image_button(curr["nobg_data"], f"nobg_{selected_file_name}")

            st.divider()
            if final_api_key:
                col_left, col_right = st.columns([1, 2])
                
                with col_left:
                    if st.button("🪄 1. 分析場景 (Analyze)", type="primary", use_container_width=True):
                        try:
                            with st.spinner(f"分析中..."):
                                st.session_state.prompts[selected_file_name] = analyze_image_with_gemini(final_api_key, nobg_pil, current_text_model)
                        except Exception as e: st.error(str(e))

                    sel_prompt = None
                    if selected_file_name in st.session_state.prompts:
                        prompts = st.session_state.prompts[selected_file_name]
                        
                        # [關鍵修復] 安全過濾，確保資料格式正確
                        safe_prompts = [p for p in prompts if isinstance(p, dict) and 'title' in p]
                        
                        if safe_prompts:
                            title = st.radio("推薦風格:", [p["title"] for p in safe_prompts])
                            sel_prompt = next((p for p in safe_prompts if p["title"] == title), None)
                            if sel_prompt:
                                # [關鍵修復] 使用 .get() 避免 KeyError
                                reason_text = sel_prompt.get('reason', '(AI 未提供詳細說明)')
                                st.info(reason_text)
                                with st.expander("查看 Prompt"): 
                                    prompt_text = sel_prompt.get('prompt', '')
                                    # 這裡使用 st.code，它是 Streamlit 內建最穩定的複製方案
                                    st.code(prompt_text, language='text') 
                        else:
                            st.warning("AI 回傳的分析資料格式異常，請重試。")

                with col_right:
                    if sel_prompt:
                        st.markdown("#### 🛠️ 2. 生成設定")
                        
                        model_options = {PRO_IMAGE_MODEL: "🚀 Pro (高畫質/預設)", FLASH_IMAGE_MODEL: "⚡ Flash (快速)"}
                        selected_gen_model_key = st.selectbox("選擇生成模型", list(model_options.keys()), format_func=lambda x: model_options[x], index=0)
                        
                        if selected_gen_model_key == PRO_IMAGE_MODEL and st.session_state.user_model_tier != "PRO":
                            st.warning("⚠️ 您的 Key 可能僅支援 Flash，若 Pro 失敗將自動降級。")

                        extra = st.text_area("自訂額外提示詞", placeholder="例如: Add a human hand...")
                        ref_file = st.file_uploader("參考圖片", type=['png', 'jpg', 'jpeg'])
                        
                        ref_img = None
                        if ref_file:
                            ref_img = Image.open(ref_file)
                            if max(ref_img.size) > 1024: ref_img.thumbnail((1024, 1024))
                        
                        if st.button(f"🎨 3. 開始生成：{sel_prompt['title']}", type="primary", use_container_width=True):
                            try:
                                with st.spinner("生成中..."):
                                    img = generate_image_with_gemini(
                                        final_api_key, nobg_pil, sel_prompt["prompt"], 
                                        selected_gen_model_key, extra, ref_img
                                    )
                                    if selected_file_name not in st.session_state.generated_results:
                                        st.session_state.generated_results[selected_file_name] = []
                                    st.session_state.generated_results[selected_file_name].insert(0, img)
                                    gc.collect()
                            except Exception as e: st.error(str(e))
                    
                    if selected_file_name in st.session_state.generated_results:
                        st.markdown("#### 🖼️ 生成結果")
                        for i, img in enumerate(st.session_state.generated_results[selected_file_name]):
                            caption_text = f"Result #{len(st.session_state.generated_results[selected_file_name])-i}"
                            st.image(img, caption=caption_text, use_container_width=True)
                            
                            img_native = pil_to_bytes(img, "PNG")
                            img_upscaled = pil_to_bytes(upscale_image(img, 2), "PNG")
                            
                            c_btn1, c_btn2, c_btn3 = st.columns([1, 1, 1])
                            with c_btn1: st.download_button("⬇️ 原圖", img_native, f"gen_{i}_native.png", "image/png", use_container_width=True)
                            with c_btn2: st.download_button("🔍 放大(2x)", img_upscaled, f"gen_{i}_upscaled.png", "image/png", use_container_width=True)
                            with c_btn3: copy_image_button(img_native, f"gen_{selected_file_name}_{i}")
                            st.divider()
            else:
                st.info("👈 請輸入 API Key 以使用 AI 功能")
