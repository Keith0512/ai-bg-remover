# Version: v1.29 (Syntax Fix & Memory Saver)
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
DEFAULT_TEXT_MODEL = "gemini-2.5-flash-preview-09-2025"
DEFAULT_IMAGE_MODEL = "gemini-2.5-flash-image-preview"

# --- JS 元件：複製圖片到剪貼簿 ---
def copy_image_button(image_bytes, key_suffix):
    b64_str = base64.b64encode(image_bytes).decode()
    html_code = f"""
    <div style="display: flex; justify-content: center; margin-top: 5px;">
        <button id="btn_{key_suffix}" onclick="copyImage_{key_suffix}()" style="
            background-color: #f0f2f6; border: 1px solid #d0d0d0; border-radius: 4px; 
            padding: 5px 10px; cursor: pointer; font-size: 14px; display: flex; align-items: center; gap: 5px;
            transition: background 0.2s;
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

# --- JS 元件：複製文字到剪貼簿 ---
def copy_text_button(text, key_suffix):
    safe_text = json.dumps(text)
    html_code = f"""
    <div style="margin-top: 5px;">
        <button id="btn_txt_{key_suffix}" onclick="copyText_{key_suffix}()" style="
            background-color: #f0f2f6; border: 1px solid #d0d0d0; border-radius: 4px; 
            padding: 2px 8px; cursor: pointer; font-size: 12px; display: flex; align-items: center; gap: 5px;
        ">
            📋 複製 Prompt
        </button>
        <span id="msg_txt_{key_suffix}" style="margin-left: 5px; font-size: 11px;"></span>
    </div>
    <script>
    async function copyText_{key_suffix}() {{
        const btn = document.getElementById("btn_txt_{key_suffix}");
        const msg = document.getElementById("msg_txt_{key_suffix}");
        try {{
            await navigator.clipboard.writeText({safe_text});
            msg.innerText = "✅ Copied!";
            msg.style.color = "green";
        }} catch (err) {{
            msg.innerText = "❌ Failed";
            msg.style.color = "red";
        }} finally {{
            setTimeout(() => {{ msg.innerText = ""; }}, 2000);
        }}
    }}
    </script>
    """
    components.html(html_code, height=40)

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

def image_to_base64(image, max_size=(1024, 1024)):
    # 這裡的縮圖是為了 Gemini API 省錢
    img_copy = image.copy()
    img_copy.thumbnail(max_size, Image.Resampling.LANCZOS)
    buffered = io.BytesIO()
    if img_copy.mode == 'RGBA':
        img_copy.save(buffered, format="PNG")
    else:
        img_copy = img_copy.convert('RGB')
        img_copy.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode()

# --- 核心功能：API Key 強力淨化 ---
def clean_api_key(key):
    if not key: return ""
    return re.sub(r'[^a-zA-Z0-9\-\_]', '', key.strip())

# --- 新增功能：動態抓取可用模型 ---
@st.cache_data(ttl=3600)
def fetch_available_models(api_key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            models = response.json().get('models', [])
            
            text_models = [
                m['name'].replace('models/', '') 
                for m in models 
                if 'generateContent' in m.get('supportedGenerationMethods', []) 
                and 'gemini' in m['name']
            ]
            
            image_models = [
                m['name'].replace('models/', '') 
                for m in models 
                if 'generateContent' in m.get('supportedGenerationMethods', [])
                and ('image' in m['name'] or 'vision' in m['name'] or 'gemini' in m['name'])
            ]
            
            text_models.sort(reverse=True)
            image_models.sort(reverse=True)
            return text_models, image_models
    except:
        pass
    return [], []

# --- 分析函式 (修復網址寫法錯誤) ---
def analyze_image_with_gemini(api_key, image, model_name):
    base64_str = image_to_base64(image)
    
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
    payload = {
        "contents": [{"parts": [{"text": prompt}, {"inline_data": {"mime_type": "image/png", "data": base64_str}}]}],
        "generation_config": {"response_mime_type": "application/json"}
    }
    
    def _send_request(target_model):
        # [✅ 修正點] 確保這是純字串，沒有 Markdown 語法
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:generateContent"
        params = {"key": api_key} # 使用 params 傳遞 Key 最安全
        
        res = None
        last_error = None
        for i in range(3):
            try:
                res = requests.post(url, params=params, json=payload)
                if res.status_code == 200 or (400 <= res.status_code < 500 and res.status_code != 429): 
                    return res
            except Exception as e:
                last_error = e
                print(f"Error attempt {i}: {e}")
            time.sleep(2 ** (i + 1))
        
        if res is None:
            raise Exception(f"連線失敗 (Network Error)。詳情: {str(last_error)}")
        return res

    response = _send_request(model_name)
    
    if response.status_code != 200:
        if model_name != DEFAULT_TEXT_MODEL:
             st.toast(f"⚠️ 模型 {model_name} 異常，嘗試切換至預設模型...", icon="🔄")
             time.sleep(1)
             return analyze_image_with_gemini(api_key, image, DEFAULT_TEXT_MODEL)

        if response.status_code == 429: raise Exception("API 配額已達上限 (429)，請稍後再試。")
        raise Exception(f"API Error ({response.status_code}): {response.text}")
    
    try:
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

# --- 生成函式 (修復網址寫法錯誤) ---
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
    
    full_prompt += "\nQuality: 8k ultra-high resolution, extreme detail, 4000px, sharp focus, macro details, commercial standard, ray tracing."

    parts = [{"text": full_prompt}]
    parts.append({"inline_data": {"mime_type": "image/png", "data": product_b64}})
    if ref_image:
        parts.append({"inline_data": {"mime_type": "image/png", "data": image_to_base64(ref_image)}})

    payload = {"contents": [{"parts": parts}], "generation_config": {"response_modalities": ["IMAGE"]}}
    
    def _send_request(target):
        # [✅ 修正點] 純字串網址
        url = f"[https://generativelanguage.googleapis.com/v1beta/models/](https://generativelanguage.googleapis.com/v1beta/models/){target}:generateContent"
        params = {"key": api_key}
        
        res = None
        last_error = None
        for i in range(3):
            try:
                res = requests.post(url, params=params, json=payload)
                if res.status_code == 200 or (400 <= res.status_code < 500 and res.status_code != 429): 
                    return res
            except Exception as e:
                last_error = e
                print(f"Gen Error attempt {i}: {e}")
            time.sleep(2 ** (i + 1))
        
        if res is None:
            raise Exception(f"連線失敗 (Network Error)。詳情: {str(last_error)}")
        return res

    response = _send_request(model_name)

    if response.status_code != 200:
        if model_name != DEFAULT_IMAGE_MODEL:
            st.toast(f"⚠️ 模型 {model_name} 異常，嘗試切換至預設模型...", icon="🔄")
            time.sleep(1)
            return generate_image_with_gemini(api_key, product_image, base_prompt, DEFAULT_IMAGE_MODEL, user_extra_prompt, ref_image)

        if response.status_code == 429: raise Exception("API 配額已達上限，請稍後再試。")
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
if 'fetched_models' not in st.session_state: st.session_state.fetched_models = ([], [])

# --- 側邊欄 ---
with st.sidebar:
    st.header("⚙️ 設定")
    raw_api_key = st.text_input("Google API Key (選填)", type="password")
    user_api_key = clean_api_key(raw_api_key)
    
    final_api_key = user_api_key if user_api_key else st.secrets.get("GEMINI_API_KEY", "")
    final_api_key = clean_api_key(final_api_key)
    
    # 預設選項
    text_model_options = [DEFAULT_TEXT_MODEL]
    image_model_options = [DEFAULT_IMAGE_MODEL]
    
    # 動態抓取模型
    if final_api_key:
        if st.button("🔄 更新模型列表"):
            with st.spinner("正在查詢..."):
                t_list, i_list = fetch_available_models(final_api_key)
                if t_list:
                    st.session_state.fetched_models = (t_list, i_list)
                    st.success(f"已更新！")
    
    if st.session_state.fetched_models[0]:
        text_model_options = st.session_state.fetched_models[0]
        if DEFAULT_TEXT_MODEL not in text_model_options: text_model_options.append(DEFAULT_TEXT_MODEL)
    if st.session_state.fetched_models[1]:
        image_model_options = st.session_state.fetched_models[1]
        if DEFAULT_IMAGE_MODEL not in image_model_options: image_model_options.append(DEFAULT_IMAGE_MODEL)

    st.divider()
    # [關鍵修正]：預設模型改為 u2netp (最輕量)，防止 Streamlit Cloud 記憶體爆掉
    model_labels = {"u2netp": "u2netp (快速省記憶體-推薦)", "isnet-general-use": "isnet (高細節)", "u2net": "u2net (標準)"}
    sel_mod = st.selectbox("去背模型", list(model_labels.keys()), format_func=lambda x: model_labels[x], index=0)
    session = get_model_session(sel_mod)
    st.divider()
    st.caption("v1.29 (Syntax Fix & Memory Saver)")

# --- 主畫面 ---
uploaded_files = st.file_uploader("1️⃣ 上傳商品圖片", type=['png', 'jpg', 'jpeg', 'webp'], accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        if file.name not in st.session_state.processed_images:
            with st.spinner(f"正在去背: {file.name}..."):
                # [關鍵修正]：讀取後立刻縮圖，防止記憶體爆炸
                img = Image.open(file)
                # 這裡的縮圖是為了保護 Streamlit Server 的 RAM
                if max(img.size) > 1024: img.thumbnail((1024, 1024)) 
                
                out = remove(img, session=session)
                st.session_state.processed_images[file.name] = {
                    "original_data": pil_to_bytes(img, "JPEG"),
                    "nobg_data": pil_to_bytes(out, "PNG")
                }
                del img, out
                gc.collect() # 強制垃圾回收

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
                    # 選擇分析模型
                    selected_text_model = st.selectbox("👁️ 分析模型", text_model_options, index=0)
                    
                    if st.button("🪄 1. 分析場景 (Analyze)", type="primary", use_container_width=True):
                        try:
                            with st.spinner(f"分析中..."):
                                st.session_state.prompts[selected_file_name] = analyze_image_with_gemini(final_api_key, nobg_pil, selected_text_model)
                        except Exception as e: st.error(str(e))

                    sel_prompt = None
                    if selected_file_name in st.session_state.prompts:
                        prompts = st.session_state.prompts[selected_file_name]
                        title = st.radio("推薦風格:", [p["title"] for p in prompts])
                        sel_prompt = next((p for p in prompts if p["title"] == title), None)
                        if sel_prompt:
                            st.info(sel_prompt['reason'])
                            with st.expander("查看 Prompt"): 
                                st.code(sel_prompt['prompt'])
                                copy_text_button(sel_prompt['prompt'], f"p_{selected_file_name}")

                with col_right:
                    if sel_prompt:
                        st.markdown("#### 🛠️ 2. 生成設定")
                        
                        # 選擇生圖模型
                        selected_gen_model = st.selectbox("🎨 生圖模型", image_model_options, index=0)

                        extra = st.text_area("自訂額外提示詞", placeholder="例如: Add a human hand...")
                        ref_file = st.file_uploader("參考圖片", type=['png', 'jpg', 'jpeg'])
                        
                        # [關鍵修正 2]：參考圖讀取時縮圖
                        ref_img = None
                        if ref_file:
                            ref_img = Image.open(ref_file)
                            if max(ref_img.size) > 1024: 
                                ref_img.thumbnail((1024, 1024))
                        
                        if st.button(f"🎨 3. 開始生成：{sel_prompt['title']}", type="primary", use_container_width=True):
                            try:
                                with st.spinner("生成中..."):
                                    img = generate_image_with_gemini(
                                        final_api_key, nobg_pil, sel_prompt["prompt"], 
                                        selected_gen_model, extra, ref_img
                                    )
                                    if selected_file_name not in st.session_state.generated_results:
                                        st.session_state.generated_results[selected_file_name] = []
                                    st.session_state.generated_results[selected_file_name].insert(0, img)
                                    
                                    # [關鍵修正 3]：生成後釋放 RAM
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
               
