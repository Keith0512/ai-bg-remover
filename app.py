import streamlit as st
from rembg import remove, new_session
from PIL import Image
import io
import zipfile
import time
import requests
import json
import base64
import gc
import re
import streamlit.components.v1 as components

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

# --- JS 元件：複製圖片到剪貼簿 (保留好用功能) ---
def copy_image_button(image_bytes, key_suffix):
    b64_str = base64.b64encode(image_bytes).decode()
    html_code = f"""
    <div style="display: flex; justify-content: center; margin-top: 5px;">
        <button id="btn_{key_suffix}" onclick="copyImage_{key_suffix}()" style="
            background-color: #f0f2f6; border: 1px solid #d0d0d0; border-radius: 4px; 
            padding: 5px 10px; cursor: pointer; font-size: 14px; display: flex; align-items: center; gap: 5px;
        ">
            📋 複製圖片
        </button>
        <span id="msg_{key_suffix}" style="margin-left: 10px; font-size: 12px; align-self: center;"></span>
    </div>
    <script>
    async function copyImage_{key_suffix}() {{
        const btn = document.getElementById("btn_{key_suffix}");
        const msg = document.getElementById("msg_{key_suffix}");
        btn.style.backgroundColor = "#e0e0e0";
        msg.innerText = "⏳...";
        try {{
            if (!navigator.clipboard || !navigator.clipboard.write) {{ throw new Error("不支援"); }}
            const response = await fetch("data:image/png;base64,{b64_str}");
            const blob = await response.blob();
            const item = new ClipboardItem({{ "image/png": blob }});
            await navigator.clipboard.write([item]);
            msg.innerText = "✅ 已複製！";
            msg.style.color = "green";
        }} catch (err) {{
            msg.innerText = "❌ 失敗";
            msg.style.color = "red";
        }} finally {{
            setTimeout(() => {{ 
                btn.style.backgroundColor = "#f0f2f6"; 
                if(msg.innerText.includes("已複製")) msg.innerText = "";
            }}, 2000);
        }}
    }}
    </script>
    """
    components.html(html_code, height=50)

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

# --- 高品質放大函式 (保留好用功能) ---
def upscale_image(image, scale_factor=2):
    """使用 Lanczos 演算法進行高品質放大"""
    new_size = (int(image.width * scale_factor), int(image.height * scale_factor))
    return image.resize(new_size, Image.Resampling.LANCZOS)

def image_to_base64(image, max_size=(1024, 1024)):
    img_copy = image.copy()
    img_copy.thumbnail(max_size, Image.Resampling.LANCZOS)
    buffered = io.BytesIO()
    if img_copy.mode == 'RGBA':
        img_copy.save(buffered, format="PNG")
    else:
        img_copy = img_copy.convert('RGB')
        img_copy.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode()

# --- API Key 淨化 (簡單版) ---
def clean_api_key(key):
    if not key: return ""
    return key.strip().replace(" ", "").replace("\n", "").replace("\r", "")

# --- 核心功能：驗證 API Key (回歸 v1.4 的簡單邏輯) ---
def check_pro_model_access(api_key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{PRO_TEXT_MODEL}:generateContent?key={api_key}"
    payload = {"contents": [{"parts": [{"text": "Ping"}]}], "generation_config": {"max_output_tokens": 1}}
    try:
        # 完全移除 timeout 參數，模仿 v1.4
        return requests.post(url, json=payload).status_code == 200 
    except: return False

# --- 分析函式 (採用 v1.4 網路架構 + v1.8 強制 Prompt) ---
def analyze_image_with_gemini(api_key, image, model_name):
    base64_str = image_to_base64(image)
    
    prompt = """
    你是一位專業的電商視覺總監。
    請分析這張已經去背的商品圖片，並構思 4 個能大幅提升轉化率的「高階商品攝影場景」。
    請回傳一個純 JSON Array (不要 Markdown)，格式如下：
    [ { "title": "風格標題", "prompt": "詳細的英文生圖提示詞...", "reason": "為什麼適合此商品" }, ... ]
    
    設計方向：
    1. 極簡高奢 (Minimalist High-End)
    2. 真實生活感 (Authentic Lifestyle)
    3. 幾何藝術 (Abstract Geometric)
    4. 自然有機 (Nature & Organic)
    
    【重要指令】：
    所有的 prompt 結尾必須強制包含以下高品質關鍵詞：
    "High resolution, 8k, extreme detail, product photography masterpiece, sharp focus, professional lighting, cinematic composition"
    """
    payload = {
        "contents": [{"parts": [{"text": prompt}, {"inline_data": {"mime_type": "image/png", "data": base64_str}}]}],
        "generation_config": {"response_mime_type": "application/json"}
    }
    
    # [關鍵復原]：這段完全照搬 v1.4 的寫法
    def _send_request(target_model):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:generateContent?key={api_key}"
        for i in range(3):
            try:
                # 這裡不設 timeout，跟 v1.4 一樣
                res = requests.post(url, json=payload)
                if res.status_code == 200 or (400 <= res.status_code < 500 and res.status_code != 429): 
                    return res
            except requests.exceptions.RequestException:
                pass 
            time.sleep(2 ** (i + 1))
        # 這裡為了避免 UnboundLocalError，我們補強一點點
        try: return requests.post(url, json=payload)
        except: raise Exception("連線失敗，請檢查網路。")

    response = _send_request(model_name)
    
    # 降級邏輯
    if response.status_code != 200 and model_name == PRO_TEXT_MODEL:
        st.toast(f"⚠️ Pro 模型異常，自動降級...", icon="🔄")
        time.sleep(1)
        response = _send_request(FLASH_TEXT_MODEL)
    
    if response.status_code != 200:
        if response.status_code == 429: raise Exception("API 配額已達上限 (429)。")
        raise Exception(f"API Error ({response.status_code}): {response.text}")
    
    try:
        # 解析邏輯維持 v1.8 的強固版
        data = response.json()
        if 'candidates' not in data: raise Exception("No candidates")
        cand = data['candidates'][0]
        if cand.get('finishReason') == 'SAFETY': raise Exception("Safety Block")
        parts = cand.get('content', {}).get('parts', [])
        if not parts: raise Exception("No parts")
        
        text_content = parts[0]['text']
        if text_content.startswith("```json"):
            text_content = text_content.replace("```json", "").replace("```", "")
            
        return json.loads(text_content)
    except Exception as e:
        raise Exception(f"解析失敗: {str(e)}")

# --- 生成函式 (採用 v1.4 網路架構 + v1.8 畫質功能) ---
def generate_image_with_gemini(api_key, product_image, base_prompt, model_name, user_extra_prompt="", ref_image=None):
    product_b64 = image_to_base64(product_image)
    
    full_prompt = f"""
    Professional product photography masterpiece.
    Subject: The FIRST image provided is the PRODUCT. KEEP THE PRODUCT APPEARANCE EXACTLY AS IS.
    """
    if ref_image:
        full_prompt += "\nReference: The SECOND image provided is a STYLE/CHARACTER REFERENCE. Integrate the product into a scene consistent with this reference."
    
    full_prompt += f"\nBackground & Atmosphere: {base_prompt}"
    if user_extra_prompt:
        full_prompt += f"\nAdditional User Requirements: {user_extra_prompt}"
    
    # v1.8 特色：強制全開 8K 畫質
    full_prompt += "\nQuality: 8k ultra-high resolution, extreme detail, 4000px, sharp focus, macro details, commercial standard, ray tracing."

    parts = [{"text": full_prompt}]
    parts.append({"inline_data": {"mime_type": "image/png", "data": product_b64}})
    if ref_image:
        parts.append({"inline_data": {"mime_type": "image/png", "data": image_to_base64(ref_image)}})

    payload = {"contents": [{"parts": parts}], "generation_config": {"response_modalities": ["IMAGE"]}}
    
    # [關鍵復原]：這段完全照搬 v1.4 的寫法
    def _send_request(target):
        url = f"[https://generativelanguage.googleapis.com/v1beta/models/](https://generativelanguage.googleapis.com/v1beta/models/){target}:generateContent?key={api_key}"
        for i in range(3):
            try:
                # 不設 timeout
                res = requests.post(url, json=payload)
                if res.status_code == 200 or (400 <= res.status_code < 500 and res.status_code != 429): 
                    return res
            except requests.exceptions.RequestException:
                pass
            time.sleep(2 ** (i + 1))
        # 最後一搏
        try: return requests.post(url, json=payload)
        except: raise Exception("連線失敗，請檢查網路。")

    response = _send_request(model_name)

    if response.status_code != 200 and "pro" in model_name:
        st.toast(f"⚠️ Pro 模型異常，自動切換至 Flash...", icon="🔄")
        time.sleep(1)
        response = _send_request(FLASH_IMAGE_MODEL)
    
    if response.status_code != 200:
        if response.status_code == 429: raise Exception("API 配額已達上限。")
        raise Exception(f"API Error ({response.status_code}): {response.text}")
        
    try:
        cand = response.json().get('candidates', [{}])[0]
        if cand.get('finishReason') == 'SAFETY': raise Exception("圖片生成因安全政策被攔截。")
        inline_data = cand.get('content', {}).get('parts', [{}])[0].get('inlineData', {})
        if not inline_data: raise Exception("模型未回傳圖片數據。")
        
        return Image.open(io.BytesIO(base64.b64decode(inline_data.get('data'))))

    except Exception as e:
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
    model_labels = {"isnet-general-use": "isnet (推薦)", "u2net": "u2net (標準)", "u2netp": "u2netp (快速)"}
    sel_mod = st.selectbox("去背模型", list(model_labels.keys()), format_func=lambda x: model_labels[x])
    session = get_model_session(sel_mod)
    st.divider()
    st.caption("v1.18 (Best of Both Worlds: v1.4 Logic + v1.8 Features)")

# --- 主畫面 ---
uploaded_files = st.file_uploader("1️⃣ 上傳商品圖片", type=['png', 'jpg', 'jpeg', 'webp'], accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        if file.name not in st.session_state.processed_images:
            with st.spinner(f"正在去背: {file.name}..."):
                img = Image.open(file)
                if max(img.size) > 2048: img.thumbnail((2048, 2048))
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
                        title = st.radio("推薦風格:", [p["title"] for p in prompts])
                        sel_prompt = next((p for p in prompts if p["title"] == title), None)
                        if sel_prompt:
                            st.info(sel_prompt['reason'])
                            with st.expander("查看 Prompt"): st.code(sel_prompt['prompt'])

                with col_right:
                    if sel_prompt:
                        st.markdown("#### 🛠️ 2. 生成設定")
                        
                        model_options = {FLASH_IMAGE_MODEL: "⚡ Flash (快速)", PRO_IMAGE_MODEL: "🚀 Pro (高畫質)"}
                        selected_gen_model_key = st.selectbox("選擇生成模型", list(model_options.keys()), format_func=lambda x: model_options[x], index=0)
                        
                        if selected_gen_model_key == PRO_IMAGE_MODEL and st.session_state.user_model_tier != "PRO":
                            st.warning("⚠️ 您的 Key 可能僅支援 Flash，若 Pro 失敗將自動降級。")

                        extra = st.text_area("自訂額外提示詞", placeholder="例如: Add a human hand...")
                        ref_file = st.file_uploader("參考圖片", type=['png', 'jpg', 'jpeg'])
                        ref_img = Image.open(ref_file) if ref_file else None
                        
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
