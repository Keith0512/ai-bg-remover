import streamlit as st
from rembg import remove, new_session
from PIL import Image
import io
import zipfile
import time
import requests
import json
import base64
import gc  # 記憶體回收機制

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

# --- 記憶體優化輔助函式 ---
def pil_to_bytes(image, format="PNG", quality=85):
    """將 PIL 圖片轉為 Bytes 以節省 Session State 記憶體"""
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
    """從 Bytes 還原為 PIL 圖片"""
    return Image.open(io.BytesIO(image_bytes))

# --- 關鍵防護：強制縮圖以節省 Token ---
def image_to_base64(image, max_size=(1024, 1024)):
    """
    將圖片轉為 Base64，並限制最大尺寸。
    🛡️ 保護機制：無論上傳多大的圖，都會在此被攔截並縮小，防止 API 費用暴增。
    """
    img_copy = image.copy()
    img_copy.thumbnail(max_size, Image.Resampling.LANCZOS)
    
    buffered = io.BytesIO()
    if img_copy.mode == 'RGBA':
        img_copy.save(buffered, format="PNG")
    else:
        img_copy = img_copy.convert('RGB')
        img_copy.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode()

# --- 核心功能：驗證 API Key 權限 ---
def check_pro_model_access(api_key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{PRO_TEXT_MODEL}:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": "Ping"}]}],
        "generation_config": {"max_output_tokens": 1}
    }
    try:
        response = requests.post(url, json=payload)
        return response.status_code == 200
    except:
        return False

# --- 輔助函式：呼叫 Gemini API (分析) ---
def analyze_image_with_gemini(api_key, image, model_name):
    # 這裡也會經過縮圖保護
    base64_str = image_to_base64(image)
    
    prompt = """
    你是一位專業的電商視覺總監。
    請分析這張已經去背的商品圖片，並構思 4 個能大幅提升轉化率的「高階商品攝影場景」。
    請回傳一個純 JSON Array (不要 Markdown)，格式如下：
    [ { "title": "風格標題", "prompt": "詳細的英文生圖提示詞...", "reason": "為什麼適合此商品" }, ... ]
    設計方向：極簡高奢、真實生活感、幾何藝術、自然有機。
    Prompt 必須是英文，強調 "High resolution, 8k, product photography masterpiece"。
    """
    
    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/png", "data": base64_str}}
            ]
        }],
        "generation_config": {"response_mime_type": "application/json"}
    }
    
    def _send_request(target_model):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:generateContent?key={api_key}"
        for i in range(3):
            try:
                res = requests.post(url, json=payload)
                if res.status_code == 200 or (400 <= res.status_code < 500 and res.status_code != 429):
                    return res
            except requests.exceptions.RequestException:
                pass 
            time.sleep(2 ** (i + 1))
        return res

    response = _send_request(model_name)
    
    if response.status_code != 200 and model_name == PRO_TEXT_MODEL:
        st.toast(f"⚠️ Pro 模型 ({model_name}) 異常，切換至 Flash 重試...", icon="🔄")
        time.sleep(1)
        response = _send_request(FLASH_TEXT_MODEL)
    
    if response.status_code != 200:
        if response.status_code == 429:
            raise Exception("API 配額已達上限 (429)，請稍後再試。")
        raise Exception(f"API Error: {response.text}")
    
    try:
        response_json = response.json()
        if 'candidates' not in response_json or not response_json['candidates']:
             if 'promptFeedback' in response_json:
                 block_reason = response_json['promptFeedback'].get('blockReason')
                 if block_reason: raise Exception(f"Prompt 被攔截: {block_reason}")
             raise Exception("模型未回傳結果 (No candidates)。")
        
        candidate = response_json['candidates'][0]
        if candidate.get('finishReason') == 'SAFETY':
             raise Exception("分析內容因安全政策被攔截，請嘗試更換圖片或模型。")
             
        parts = candidate.get('content', {}).get('parts', [])
        if not parts:
            raise Exception("模型回傳內容缺少 'parts' 欄位。")
            
        return json.loads(parts[0]['text'])
    except Exception as e:
        raise Exception(f"解析分析結果失敗: {str(e)}")

# --- 輔助函式：呼叫 Gemini API (生成) ---
def generate_image_with_gemini(api_key, product_image, base_prompt, model_name, user_extra_prompt="", ref_image=None):
    # 1. 商品圖：經過縮圖保護
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
        
    full_prompt += "\nQuality: 8k resolution, highly detailed, commercial advertisement standard."

    parts = [{"text": full_prompt}]
    parts.append({"inline_data": {"mime_type": "image/png", "data": product_b64}})
    
    if ref_image:
        # 2. 參考圖：✅ 這裡同樣呼叫了 image_to_base64，所以絕對有縮圖保護
        ref_b64 = image_to_base64(ref_image)
        parts.append({"inline_data": {"mime_type": "image/png", "data": ref_b64}})

    payload = {
        "contents": [{"parts": parts}],
        "generation_config": {"response_modalities": ["IMAGE"]}
    }
    
    def _send_request(target_model):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:generateContent?key={api_key}"
        for i in range(3):
            try:
                res = requests.post(url, json=payload)
                if res.status_code == 200 or (400 <= res.status_code < 500 and res.status_code != 429):
                    return res
            except requests.exceptions.RequestException:
                pass
            time.sleep(2 ** (i + 1))
        return res

    response = _send_request(model_name)

    if response.status_code != 200 and model_name == PRO_IMAGE_MODEL:
        st.toast(f"⚠️ Pro 生圖模型 ({model_name}) 異常，切換至 Flash 重試...", icon="🔄")
        time.sleep(1)
        response = _send_request(FLASH_IMAGE_MODEL)
    
    if response.status_code != 200:
        if response.status_code == 429:
            raise Exception("API 配額已達上限 (429)，請稍後再試。")
        raise Exception(f"API Error: {response.text}")
        
    try:
        response_json = response.json()
        if 'candidates' not in response_json or not response_json['candidates']:
             if 'promptFeedback' in response_json:
                 block_reason = response_json['promptFeedback'].get('blockReason')
                 if block_reason: raise Exception(f"Prompt 被攔截: {block_reason}")
             raise Exception("模型未回傳結果。")

        candidate = response_json['candidates'][0]
        if candidate.get('finishReason') == 'SAFETY':
             raise Exception("圖片生成因安全政策被攔截，請調整風格描述。")

        parts = candidate.get('content', {}).get('parts', [])
        if not parts: raise Exception("內容為空。")
             
        part = parts[0]
        inline_data = part.get('inlineData') or part.get('inline_data')
        
        if inline_data:
            img_b64 = inline_data.get('data')
            if img_b64: return Image.open(io.BytesIO(base64.b64decode(img_b64)))
        
        if part.get('text'):
             raise Exception(f"模型回傳了文字而非圖片: {part.get('text')[:50]}...")
             
        raise Exception(f"無法解析圖片數據。")

    except Exception as e:
        raise Exception(f"生成失敗: {str(e)}")

# --- 快取模型 Session ---
@st.cache_resource
def get_model_session(model_name):
    return new_session(model_name)

# --- 主標題區 ---
st.title("🛍️ AI 電商圖一條龍生成器")
st.markdown(f"""
結合 **rembg** 與 **Gemini** 生成能力。
預設使用 **Flash ({FLASH_TEXT_MODEL})**，輸入綁定帳單的 API Key 可解鎖 **Pro** 模型。
""")

# --- Session State 初始化 ---
if 'processed_images' not in st.session_state:
    st.session_state.processed_images = {}
if 'prompts' not in st.session_state:
    st.session_state.prompts = {}
if 'generated_results' not in st.session_state:
    st.session_state.generated_results = {}
if 'last_validated_key' not in st.session_state:
    st.session_state.last_validated_key = None
if 'user_model_tier' not in st.session_state:
    st.session_state.user_model_tier = "FLASH" 

# --- 側邊欄設定 ---
with st.sidebar:
    st.header("⚙️ 設定")
    
    user_api_key = st.text_input("Google API Key (選填)", type="password", help="輸入後將自動測試是否支援 Pro 模型")
    final_api_key = user_api_key if user_api_key else st.secrets.get("GEMINI_API_KEY", "")
    
    if user_api_key and user_api_key != st.session_state.last_validated_key:
        with st.spinner("正在驗證 API Key 權限 (Gemini 3 Pro)..."):
            is_pro = check_pro_model_access(user_api_key)
            if is_pro:
                st.session_state.user_model_tier = "PRO"
                st.toast("✅ 驗證成功！已啟用 Gemini 3 Pro 模型", icon="🚀")
            else:
                st.session_state.user_model_tier = "FLASH"
                st.error("⚠️ 無法啟用 Gemini 3 Pro 模型。\n\n您的 API Key 可能未綁定帳單。系統已自動降級為 Flash 模型。")
            st.session_state.last_validated_key = user_api_key
    elif not user_api_key:
        st.session_state.user_model_tier = "FLASH"
        st.session_state.last_validated_key = None

    if st.session_state.user_model_tier == "PRO" and user_api_key:
        current_text_model = PRO_TEXT_MODEL
        current_image_model = PRO_IMAGE_MODEL
        st.success(f"🚀 **Pro Mode Activated**\nVision: {PRO_TEXT_MODEL}\nImage: {PRO_IMAGE_MODEL}")
    else:
        current_text_model = FLASH_TEXT_MODEL
        current_image_model = FLASH_IMAGE_MODEL
        status_msg = "⚡ **Flash Mode (Default)**"
        st.info(f"{status_msg}\nVision: {FLASH_TEXT_MODEL}\nImage: {FLASH_IMAGE_MODEL}")
        if user_api_key and st.session_state.user_model_tier == "FLASH":
            st.caption("ℹ️ 您目前的 Key 僅支援免費版 (Flash)")

    st.divider()
    st.subheader("去背模型")
    model_descriptions = {
        "isnet-general-use": {"label": "isnet (高細節-推薦)", "details": "適合頭髮、毛髮等複雜邊緣。"},
        "u2net": {"label": "u2net (標準)", "details": "泛用性最高。"},
        "u2netp": {"label": "u2netp (快速)", "details": "速度最快。"}
    }
    selected_model_key = st.selectbox("選擇模型", list(model_descriptions.keys()), format_func=lambda x: model_descriptions[x]["label"])
    session = get_model_session(selected_model_key)
    
    st.divider()
    st.caption("v1.4 (Final Secure)")

# --- 主邏輯：上傳區 ---
uploaded_files = st.file_uploader("1️⃣ 上傳商品圖片", type=['png', 'jpg', 'jpeg', 'webp'], accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        # 只處理尚未處理過的檔案
        if file.name not in st.session_state.processed_images:
            with st.spinner(f"正在去背: {file.name}..."):
                # 1. 讀取圖片
                input_image = Image.open(file)
                
                # 記憶體防護：如果圖片大於 2048px，先縮小
                max_dim = 2048
                if max(input_image.size) > max_dim:
                    input_image.thumbnail((max_dim, max_dim))
                
                # 2. 去背
                output_image = remove(input_image, session=session)
                
                # 3. 轉為 Bytes 存入 Session State (不存 PIL 物件)
                st.session_state.processed_images[file.name] = {
                    "original_data": pil_to_bytes(input_image, "JPEG"), # 存 JPEG 節省空間
                    "nobg_data": pil_to_bytes(output_image, "PNG")      # 存 PNG 保留透明度
                }
                
                # 4. 強制釋放記憶體
                del input_image
                del output_image
                gc.collect()

    st.divider()
    st.subheader("2️⃣ AI 分析與生成")
    
    if st.session_state.processed_images:
        selected_file_name = st.selectbox("選擇商品", list(st.session_state.processed_images.keys()))
        
        if selected_file_name:
            current_data = st.session_state.processed_images[selected_file_name]
            
            # 從 Bytes 還原 PIL 物件供顯示用 (用完即丟)
            original_pil = bytes_to_pil(current_data["original_data"])
            nobg_pil = bytes_to_pil(current_data["nobg_data"])
            
            col1, col2 = st.columns(2)
            with col1: st.image(original_pil, caption="原始", use_container_width=True)
            with col2: st.image(nobg_pil, caption="去背", use_container_width=True)
            
            st.download_button("⬇️ 下載去背圖", current_data["nobg_data"], f"{selected_file_name}_nobg.png", "image/png")

            st.divider()
            if final_api_key:
                c1, c2 = st.columns([1, 2])
                
                with c1:
                    if st.button("🪄 1. 分析場景 (Analyze)", type="primary"):
                        try:
                            with st.spinner(f"分析中 ({current_text_model})..."):
                                prompts = analyze_image_with_gemini(final_api_key, nobg_pil, current_text_model)
                                st.session_state.prompts[selected_file_name] = prompts
                        except Exception as e: st.error(str(e))

                    selected_prompt_data = None
                    if selected_file_name in st.session_state.prompts:
                        prompts = st.session_state.prompts[selected_file_name]
                        title = st.radio("選擇 AI 推薦風格:", [p["title"] for p in prompts])
                        selected_prompt_data = next((p for p in prompts if p["title"] == title), None)
                        if selected_prompt_data:
                            st.info(selected_prompt_data['reason'])
                            with st.expander("查看原始 Prompt"): st.code(selected_prompt_data['prompt'])

                with c2:
                    if selected_prompt_data:
                        st.markdown("#### 🛠️ 2. 進階設定 (Optional)")
                        user_extra_prompt = st.text_area("📝 自訂額外提示詞", placeholder="例如: Add a human hand holding the product...")
                        
                        ref_image_file = st.file_uploader("🖼️ 上傳參考圖片 (例如: 人物、風格圖)", type=['png', 'jpg', 'jpeg', 'webp'], key="ref_img")
                        ref_image = Image.open(ref_image_file) if ref_image_file else None
                        if ref_image: st.image(ref_image, caption="已載入參考圖", width=150)

                        st.markdown("---")
                        
                        if st.button(f"🎨 3. 開始生成：{selected_prompt_data['title']}", type="primary"):
                            try:
                                with st.spinner(f"生成中 ({current_image_model})..."):
                                    img = generate_image_with_gemini(
                                        api_key=final_api_key, 
                                        product_image=nobg_pil, 
                                        base_prompt=selected_prompt_data["prompt"], 
                                        model_name=current_image_model,
                                        user_extra_prompt=user_extra_prompt,
                                        ref_image=ref_image
                                    )
                                    if selected_file_name not in st.session_state.generated_results:
                                        st.session_state.generated_results[selected_file_name] = []
                                    st.session_state.generated_results[selected_file_name].insert(0, img)
                            except Exception as e: st.error(str(e))
                    
                    if selected_file_name in st.session_state.generated_results:
                        st.markdown("#### 🖼️ 生成結果")
                        for i, img in enumerate(st.session_state.generated_results[selected_file_name]):
                            st.image(img, caption=f"Result #{len(st.session_state.generated_results[selected_file_name])-i}", use_container_width=True)
                            buf = io.BytesIO()
                            img.save(buf, format='PNG')
                            st.download_button(f"⬇️ 下載", buf.getvalue(), f"gen_{i}.png", "image/png", key=f"d_{i}")
            else:
                st.info("👈 請輸入 API Key 以使用 AI 功能")
