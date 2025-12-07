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

# --- 常數設定 (預設模型) ---
DEFAULT_TEXT_MODEL = "gemini-2.5-flash-preview-09-2025"
DEFAULT_IMAGE_MODEL = "gemini-2.5-flash-image-preview"

# --- 輔助函式：圖片轉 Base64 ---
def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# --- 輔助函式：呼叫 Gemini API (分析) ---
def analyze_image_with_gemini(api_key, image, model_name):
    base64_str = image_to_base64(image)
    
    # 定義提示詞與 Payload
    prompt = """
    你是一位專業的電商視覺總監。
    請分析這張已經去背的商品圖片，並構思 4 個能大幅提升轉化率的「高階商品攝影場景」。
    
    請回傳一個純 JSON Array (不要 Markdown)，格式如下：
    [
      { "title": "風格標題", "prompt": "詳細的英文生圖提示詞...", "reason": "為什麼適合此商品" },
      ...
    ]

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
    
    # 內部函式：發送請求 (含 Retry 機制)
    def _send_request(target_model):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:generateContent?key={api_key}"
        # 最多重試 3 次 (等待 2s, 4s, 8s)
        for i in range(3):
            res = requests.post(url, json=payload)
            if res.status_code != 429:
                return res
            time.sleep(2 ** (i + 1)) # 指數退避
        return res

    # 第一次嘗試：使用指定模型 (可能是 Pro)
    response = _send_request(model_name)
    
    # 如果遇到 任何錯誤 (非200) 且當前不是預設模型，則自動降級
    if response.status_code != 200 and model_name != DEFAULT_TEXT_MODEL:
        st.toast(f"⚠️ Pro 模型 ({model_name}) 發生錯誤 (Code: {response.status_code})，自動降級至 Flash 模型...", icon="🔄")
        time.sleep(1) # 稍作緩衝
        response = _send_request(DEFAULT_TEXT_MODEL)
    
    if response.status_code != 200:
        if response.status_code == 429:
            raise Exception("API 配額已達上限 (429)。Google 免費版 API 有每分鐘請求限制，請稍等 1 分鐘後再試。")
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
    
    # 內部函式：發送請求 (含 Retry 機制)
    def _send_request(target_model):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:generateContent?key={api_key}"
        # 最多重試 3 次 (等待 2s, 4s, 8s)
        for i in range(3):
            res = requests.post(url, json=payload)
            if res.status_code != 429:
                return res
            time.sleep(2 ** (i + 1)) # 指數退避
        return res

    # 第一次嘗試
    response = _send_request(model_name)

    # 如果遇到 任何錯誤 (非200) 且當前不是預設模型 (例如是 Pro)，則自動降級
    if response.status_code != 200 and model_name != DEFAULT_IMAGE_MODEL:
        st.toast(f"⚠️ Pro 生圖模型 ({model_name}) 發生錯誤 (Code: {response.status_code})，自動切換至 Flash 模型...", icon="🔄")
        time.sleep(1)
        response = _send_request(DEFAULT_IMAGE_MODEL)
    
    if response.status_code != 200:
        if response.status_code == 429:
            raise Exception("API 配額已達上限 (429)。Google 免費版 API 有每分鐘請求限制，請稍等 1 分鐘後再試。")
        raise Exception(f"API Error: {response.text}")
        
    # 解析回傳的圖片
    try:
        response_json = response.json()
        
        # 檢查是否有候選結果
        if 'candidates' not in response_json or not response_json['candidates']:
             # 有時 API 雖然 200 OK 但沒有 candidates (例如被過濾)
             if 'promptFeedback' in response_json:
                 block_reason = response_json['promptFeedback'].get('blockReason')
                 if block_reason:
                     raise Exception(f"Prompt 被系統攔截: {block_reason}")
             raise Exception("模型未回傳任何候選結果 (No candidates returned)。")

        candidate = response_json['candidates'][0]
        
        # 檢查是否因為安全原因結束
        if candidate.get('finishReason') == 'SAFETY':
             safety_ratings = candidate.get('safetyRatings', [])
             # 簡單列出觸發的安全類別
             reasons = [r['category'] for r in safety_ratings if r['probability'] in ['MEDIUM', 'HIGH']]
             raise Exception(f"圖片生成因「安全政策」被攔截。觸發類別: {', '.join(reasons)}。請嘗試調整風格描述。")

        parts = candidate.get('content', {}).get('parts', [])
        if not parts:
             raise Exception("模型回傳內容為空。")
             
        # 嘗試取得 inlineData (REST API 標準) 或 inline_data (相容舊版/SDK)
        part = parts[0]
        inline_data = part.get('inlineData') or part.get('inline_data')
        
        if inline_data:
            img_b64 = inline_data.get('data')
            if img_b64:
                return Image.open(io.BytesIO(base64.b64decode(img_b64)))
        
        # 如果沒有圖片數據，檢查是否有文字錯誤訊息
        if part.get('text'):
             raise Exception(f"模型回傳了文字而非圖片: '{part.get('text')[:100]}...'。這表示模型拒絕生成圖片，請檢查 Prompt 或更換模型。")
             
        raise Exception(f"無法解析圖片數據，API 回傳結構異常。")

    except Exception as e:
        # 捕捉並重新拋出具體錯誤，保留原始錯誤訊息
        raise Exception(f"生成失敗: {str(e)}")

# --- 快取模型 Session ---
@st.cache_resource
def get_model_session(model_name):
    return new_session(model_name)

# --- 主標題區 ---
st.title("🛍️ AI 電商圖一條龍生成器")
st.markdown("""
結合 **rembg** 強大去背與 **Gemini Pro** 生成能力。
1. **去背**：上傳圖片，自動移除背景。
2. **分析**：AI 自動分析商品並推薦場景。
3. **生成**：一鍵合成高質感電商廣告圖。
""")

# --- Session State 初始化 ---
if 'processed_images' not in st.session_state:
    st.session_state.processed_images = {} # 用 dict 存，key 是檔名
if 'prompts' not in st.session_state:
    st.session_state.prompts = {}
if 'generated_results' not in st.session_state:
    st.session_state.generated_results = {}

# --- 側邊欄設定 ---
with st.sidebar:
    st.header("⚙️ 設定")
    
    # API Key 輸入與模型邏輯
    user_api_key = st.text_input("Google API Key (選填)", type="password", help="輸入 API Key 可升級至 Gemini 3 Pro 模型；未輸入則使用預設 Flash 模型")
    
    # 升級模型 (Pro)
    pro_text_model = "gemini-3-pro-preview"
    pro_image_model = "gemini-3-pro-image-preview"
    
    if user_api_key:
        current_api_key = user_api_key
        current_text_model = pro_text_model
        current_image_model = pro_image_model
        st.success(f"🚀 已嘗試啟用 Pro 模型:\nVision: {pro_text_model}\nImage: {pro_image_model}")
        st.caption("若配額不足將自動切換回 Flash 模型")
    else:
        # 嘗試從 Secrets 讀取預設 Key
        current_api_key = st.secrets.get("GEMINI_API_KEY", "")
        current_text_model = DEFAULT_TEXT_MODEL
        current_image_model = DEFAULT_IMAGE_MODEL
        
        if current_api_key:
            st.info(f"⚡ 使用預設 Flash 模型:\nVision: {DEFAULT_TEXT_MODEL}\nImage: {DEFAULT_IMAGE_MODEL}")
        else:
            st.warning("⚠️ 未偵測到預設 Key 且未輸入 API Key，生成功能可能無法使用")

    st.divider()
    st.subheader("去背模型選擇")
    
    model_descriptions = {
        "u2net": {"label": "u2net (標準通用)", "details": "泛用性最高，適合大多數情況。"},
        "isnet-general-use": {"label": "isnet (高細節)", "details": "適合頭髮、毛髮等複雜邊緣。"},
        "u2net_human_seg": {"label": "human_seg (人像)", "details": "專門處理人像。"},
        "u2netp": {"label": "u2netp (快速)", "details": "速度最快，適合低階設備。"}
    }

    selected_model_key = st.selectbox(
        "模型",
        options=list(model_descriptions.keys()),
        format_func=lambda x: model_descriptions[x]["label"],
        index=1 # 預設改為 isnet，效果較好
    )
    st.caption(model_descriptions[selected_model_key]["details"])
    
    session = get_model_session(selected_model_key)

# --- 主邏輯：上傳區 ---
uploaded_files = st.file_uploader(
    "1️⃣ 上傳商品圖片 (Step 1: Upload)", 
    type=['png', 'jpg', 'jpeg', 'webp'], 
    accept_multiple_files=True
)

if uploaded_files:
    # 這裡只做去背處理，不重複執行
    for file in uploaded_files:
        if file.name not in st.session_state.processed_images:
            with st.spinner(f"正在去背: {file.name}..."):
                input_image = Image.open(file)
                output_image = remove(input_image, session=session)
                # 存入 session state
                st.session_state.processed_images[file.name] = {
                    "original": input_image,
                    "nobg": output_image,
                    "file_obj": file
                }

    # 顯示處理列表
    st.divider()
    st.subheader("2️⃣ 圖片列表與 AI 生成 (Step 2 & 3)")
    
    # 選擇要處理的圖片 (如果是批次上傳，讓使用者選一張來生成，避免 API 爆量)
    selected_file_name = st.selectbox("選擇要進行 AI 生成的商品", list(st.session_state.processed_images.keys()))
    
    if selected_file_name:
        current_data = st.session_state.processed_images[selected_file_name]
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(current_data["original"], caption="原始圖片", use_container_width=True)
        with col2:
            st.image(current_data["nobg"], caption="去背結果", use_container_width=True)
            
        # 下載去背圖按鈕
        img_byte_arr = io.BytesIO()
        current_data["nobg"].save(img_byte_arr, format='PNG')
        st.download_button("⬇️ 下載此去背圖", img_byte_arr.getvalue(), f"{selected_file_name}_nobg.png", "image/png")

        st.divider()
        
        # --- AI 分析與生成區 ---
        if current_api_key:
            col_gen_1, col_gen_2 = st.columns([1, 2])
            
            with col_gen_1:
                st.markdown("#### AI 場景分析")
                analyze_btn = st.button("🪄 分析商品並推薦場景", key="analyze_btn", type="primary")
                
                if analyze_btn:
                    try:
                        with st.spinner(f"正在觀察商品細節 (Model: {current_text_model})..."):
                            # 傳入選擇的 Model
                            prompts = analyze_image_with_gemini(current_api_key, current_data["nobg"], current_text_model)
                            st.session_state.prompts[selected_file_name] = prompts
                    except Exception as e:
                        st.error(f"分析失敗: {str(e)}")

                # 顯示 Prompt 選項
                selected_prompt_data = None
                if selected_file_name in st.session_state.prompts:
                    prompts = st.session_state.prompts[selected_file_name]
                    
                    # 使用 Radio 或 Selectbox 讓使用者選
                    prompt_options = [p["title"] for p in prompts]
                    selected_prompt_title = st.radio("選擇一種風格:", prompt_options)
                    
                    # 找到對應的完整資料
                    selected_prompt_data = next((p for p in prompts if p["title"] == selected_prompt_title), None)
                    
                    if selected_prompt_data:
                        st.info(f"💡 設計理念: {selected_prompt_data['reason']}")
                        with st.expander("查看完整 Prompt"):
                            st.code(selected_prompt_data['prompt'])

            with col_gen_2:
                st.markdown("#### AI 最終生成")
                
                if selected_prompt_data:
                    generate_btn = st.button(f"🎨 生成：{selected_prompt_data['title']}", type="primary")
                    
                    if generate_btn:
                        try:
                            with st.spinner(f"正在佈置場景 (Model: {current_image_model})..."):
                                # 傳入選擇的 Model
                                result_img = generate_image_with_gemini(
                                    current_api_key, 
                                    current_data["nobg"], 
                                    selected_prompt_data["prompt"],
                                    current_image_model
                                )
                                # 存入結果
                                if selected_file_name not in st.session_state.generated_results:
                                    st.session_state.generated_results[selected_file_name] = []
                                st.session_state.generated_results[selected_file_name].insert(0, result_img) # 最新的放前面
                                
                        except Exception as e:
                            st.error(f"生成失敗: {str(e)}")

                # 顯示生成結果歷史
                if selected_file_name in st.session_state.generated_results:
                    results = st.session_state.generated_results[selected_file_name]
                    if results:
                        st.success("✨ 生成完成！")
                        for idx, img in enumerate(results):
                            st.image(img, caption=f"生成結果 #{len(results)-idx}", use_container_width=True)
                            
                            # 下載按鈕
                            res_byte_arr = io.BytesIO()
                            img.save(res_byte_arr, format='PNG')
                            st.download_button(
                                f"⬇️ 下載結果圖 #{len(results)-idx}", 
                                res_byte_arr.getvalue(), 
                                f"gen_{selected_file_name}_{idx}.png", 
                                "image/png",
                                key=f"dl_gen_{idx}"
                            )
                            st.divider()
        else:
            st.info("👈 請在左側設定輸入 API Key 以解鎖 AI 生成功能")

else:
    # 歡迎畫面
    st.info("請上傳圖片以開始。支援批次上傳。")
