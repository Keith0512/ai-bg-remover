import streamlit as st
from rembg import remove, new_session
from PIL import Image
import io
import zipfile
import time
import requests
import json
import base64

# --- 設定頁面資訊 ---
st.set_page_config(
    page_title="AI 電商圖一條龍生成器",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 常數設定 ---
# Pro 模型 (需要 Billing)
PRO_TEXT_MODEL = "gemini-3-pro-preview"
PRO_IMAGE_MODEL = "gemini-3-pro-image-preview"

# Flash 模型 (免費額度較高)
FLASH_TEXT_MODEL = "gemini-2.5-flash-preview-09-2025"
FLASH_IMAGE_MODEL = "gemini-2.5-flash-image-preview"

# --- 輔助函式：圖片轉 Base64 ---
def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# --- 核心功能：驗證 API Key 權限 ---
def check_pro_model_access(api_key):
    """
    發送一個極輕量的請求給 Pro 模型，測試是否可用。
    如果回傳 200，代表有權限 (有綁定帳單)；否則回傳 False。
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{PRO_TEXT_MODEL}:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": "Ping"}]}],
        "generation_config": {"max_output_tokens": 1} # 極小化 token 消耗
    }
    try:
        response = requests.post(url, json=payload)
        return response.status_code == 200
    except:
        return False

# --- 輔助函式：呼叫 Gemini API (分析) ---
def analyze_image_with_gemini(api_key, image, model_name):
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
            res = requests.post(url, json=payload)
            if res.status_code != 429: return res
            time.sleep(2 ** (i + 1))
        return res

    response = _send_request(model_name)
    
    # 雙重保險：執行期間若遇到問題，再次嘗試降級
    if response.status_code != 200 and model_name == PRO_TEXT_MODEL:
        st.toast(f"⚠️ Pro 模型 ({model_name}) 執行失敗 (Code: {response.status_code})，切換至 Flash 重試...", icon="🔄")
        time.sleep(1)
        response = _send_request(FLASH_TEXT_MODEL)
    
    if response.status_code != 200:
        if response.status_code == 429:
            raise Exception("API 配額已達上限，請稍後再試。")
        raise Exception(f"API Error: {response.text}")
        
    return json.loads(response.json()['candidates'][0]['content']['parts'][0]['text'])

# --- 輔助函式：呼叫 Gemini API (生成) ---
def generate_image_with_gemini(api_key, image, prompt_text, model_name):
    base64_str = image_to_base64(image)
    
    full_prompt = f"""
    Professional product photography masterpiece.
    Subject: The product in the reference image. KEEP THE PRODUCT EXACTLY AS IS.
    Background & Atmosphere: {prompt_text}
    Quality: 8k resolution, highly detailed, commercial advertisement standard.
    """
    
    payload = {
        "contents": [{
            "parts": [
                {"text": full_prompt},
                {"inline_data": {"mime_type": "image/png", "data": base64_str}}
            ]
        }],
        "generation_config": {"response_modalities": ["IMAGE"]}
    }
    
    def _send_request(target_model):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:generateContent?key={api_key}"
        for i in range(3):
            res = requests.post(url, json=payload)
            if res.status_code != 429: return res
            time.sleep(2 ** (i + 1))
        return res

    response = _send_request(model_name)

    # 雙重保險：執行期間若遇到問題，再次嘗試降級
    if response.status_code != 200 and model_name == PRO_IMAGE_MODEL:
        st.toast(f"⚠️ Pro 生圖模型 ({model_name}) 執行失敗，切換至 Flash 重試...", icon="🔄")
        time.sleep(1)
        response = _send_request(FLASH_IMAGE_MODEL)
    
    if response.status_code != 200:
        if response.status_code == 429:
            raise Exception("API 配額已達上限，請稍後再試。")
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
# 用來記錄上次驗證過的 Key，避免重複驗證
if 'last_validated_key' not in st.session_state:
    st.session_state.last_validated_key = None
if 'user_model_tier' not in st.session_state:
    st.session_state.user_model_tier = "FLASH" # FLASH or PRO

# --- 側邊欄設定 ---
with st.sidebar:
    st.header("⚙️ 設定")
    
    user_api_key = st.text_input("Google API Key (選填)", type="password", help="輸入後將自動測試是否支援 Pro 模型")
    
    # --- 關鍵邏輯：API Key 驗證與模型選擇 ---
    final_api_key = user_api_key if user_api_key else st.secrets.get("GEMINI_API_KEY", "")
    
    # 當 API Key 改變時，執行驗證
    if user_api_key and user_api_key != st.session_state.last_validated_key:
        with st.spinner("正在驗證 API Key 權限 (Gemini 3 Pro)..."):
            is_pro = check_pro_model_access(user_api_key)
            if is_pro:
                st.session_state.user_model_tier = "PRO"
                st.toast("✅ 驗證成功！已啟用 Gemini 3 Pro 模型", icon="🚀")
            else:
                st.session_state.user_model_tier = "FLASH"
                # 這裡顯示您要求的警告
                st.error("⚠️ 無法啟用 Gemini 3 Pro 模型。\n\n您的 API Key 可能未綁定帳單。系統已自動降級為 Flash 模型。\n\n💡 若要使用 Pro 功能，請前往 Google AI Studio 綁定信用卡/帳單。")
            st.session_state.last_validated_key = user_api_key
    elif not user_api_key:
        # 如果使用者清空 Key，重置為 Flash
        st.session_state.user_model_tier = "FLASH"
        st.session_state.last_validated_key = None

    # 根據驗證結果設定當前模型
    if st.session_state.user_model_tier == "PRO" and user_api_key:
        current_text_model = PRO_TEXT_MODEL
        current_image_model = PRO_IMAGE_MODEL
        st.success(f"🚀 **Pro Mode Activated**\nVision: {PRO_TEXT_MODEL}\nImage: {PRO_IMAGE_MODEL}")
    else:
        current_text_model = FLASH_TEXT_MODEL
        current_image_model = FLASH_IMAGE_MODEL
        
        status_msg = "⚡ **Flash Mode (Default)**"
        st.info(f"{status_msg}\nVision: {FLASH_TEXT_MODEL}\nImage: {FLASH_IMAGE_MODEL}")
        
        # 如果有輸入 Key 但不在 Pro 模式，顯示一個常駐的小提示
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

# --- 主邏輯：上傳區 ---
uploaded_files = st.file_uploader("1️⃣ 上傳商品圖片", type=['png', 'jpg', 'jpeg', 'webp'], accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        if file.name not in st.session_state.processed_images:
            with st.spinner(f"正在去背: {file.name}..."):
                input_image = Image.open(file)
                output_image = remove(input_image, session=session)
                st.session_state.processed_images[file.name] = {"original": input_image, "nobg": output_image}

    st.divider()
    st.subheader("2️⃣ AI 分析與生成")
    selected_file_name = st.selectbox("選擇商品", list(st.session_state.processed_images.keys()))
    
    if selected_file_name:
        current_data = st.session_state.processed_images[selected_file_name]
        col1, col2 = st.columns(2)
        with col1: st.image(current_data["original"], caption="原始", use_container_width=True)
        with col2: st.image(current_data["nobg"], caption="去背", use_container_width=True)
        
        # 下載去背
        buf = io.BytesIO()
        current_data["nobg"].save(buf, format='PNG')
        st.download_button("⬇️ 下載去背圖", buf.getvalue(), f"{selected_file_name}_nobg.png", "image/png")

        st.divider()
        if final_api_key:
            c1, c2 = st.columns([1, 2])
            with c1:
                if st.button("🪄 分析場景", type="primary"):
                    try:
                        with st.spinner(f"分析中 ({current_text_model})..."):
                            prompts = analyze_image_with_gemini(final_api_key, current_data["nobg"], current_text_model)
                            st.session_state.prompts[selected_file_name] = prompts
                    except Exception as e: st.error(str(e))

                selected_prompt_data = None
                if selected_file_name in st.session_state.prompts:
                    prompts = st.session_state.prompts[selected_file_name]
                    title = st.radio("風格:", [p["title"] for p in prompts])
                    selected_prompt_data = next((p for p in prompts if p["title"] == title), None)
                    if selected_prompt_data:
                        st.info(selected_prompt_data['reason'])
                        with st.expander("Prompt"): st.code(selected_prompt_data['prompt'])

            with c2:
                if selected_prompt_data and st.button(f"🎨 生成：{selected_prompt_data['title']}", type="primary"):
                    try:
                        with st.spinner(f"生成中 ({current_image_model})..."):
                            img = generate_image_with_gemini(final_api_key, current_data["nobg"], selected_prompt_data["prompt"], current_image_model)
                            if selected_file_name not in st.session_state.generated_results:
                                st.session_state.generated_results[selected_file_name] = []
                            st.session_state.generated_results[selected_file_name].insert(0, img)
                    except Exception as e: st.error(str(e))
                
                if selected_file_name in st.session_state.generated_results:
                    for i, img in enumerate(st.session_state.generated_results[selected_file_name]):
                        st.image(img, caption=f"結果 #{i+1}", use_container_width=True)
                        buf = io.BytesIO()
                        img.save(buf, format='PNG')
                        st.download_button(f"⬇️ 下載 #{i+1}", buf.getvalue(), f"gen_{i}.png", "image/png", key=f"d_{i}")
        else:
            st.info("👈 請輸入 API Key 以使用 AI 功能")
