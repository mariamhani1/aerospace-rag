import streamlit as st
import os
from qdrant_client import QdrantClient
from mlx_vlm import load, generate
from colpali_engine.models import ColPali, ColPaliProcessor
import torch

# --- Configuration ---
QDRANT_URL = "https://747b294f-4459-4b70-9beb-ffa601e0da44.europe-west3-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6MTExZTEyODgtNDAyNS00NGEyLTlhZjItMWU0NGVkNjExOGY3In0.Estw8GLjLZwIFhbhY4FOb14EMhFK4v36Pl8Jet4aLsI"
COLLECTION_NAME = "aerospace_manuals"
COLLECTION_NAME = "aerospace_manuals"

MODEL_PATH = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
COLPALI_PATH = "vidore/colpali-v1.2-merged"

# --- Initialization ---
@st.cache_resource
def init_qdrant():
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

@st.cache_resource
def init_vision_model():
    # Apple MLX handles the Vision-Language generation
    model, processor = load(MODEL_PATH)
    return model, processor

@st.cache_resource
def init_text_encoder():
    # PyTorch handles the text-to-vector embedding (Forced to CPU to save your RAM)
    processor = ColPaliProcessor.from_pretrained(COLPALI_PATH)
    model = ColPali.from_pretrained(COLPALI_PATH, torch_dtype=torch.bfloat16, device_map="cpu")
    return processor, model

client = init_qdrant()
vlm_model, vlm_processor = init_vision_model()
colpali_processor, colpali_model = init_text_encoder()

# --- Real Multi-Vector Retrieval ---
def get_relevant_image(query_text):
    # 1. Convert user text into a multi-vector query
    inputs = colpali_processor.process_queries([query_text]).to("cpu")
    with torch.no_grad():
        embeddings = colpali_model(**inputs)
    
    query_multivector = embeddings[0].float().numpy().tolist()
    
    # 2. Search Frankfurt for the exact matching visual patches
    result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_multivector,
        limit=1,
        with_payload=True
    )
    
    if not result.points:
        return None, None
        
    top_point = result.points[0]
    return top_point.payload['image_filename'], top_point.score

# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="Aerospace RAG")

st.markdown("""
<style>
    /* Aerospace Theme Styling */
    .stApp {
        background-color: #0b101e; /* Deep Space Blue */
    }
    h1, h2, h3, h4, p, label, .stMarkdown {
        color: #e2e8f0 !important; /* Star White */
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif !important;
    }
    .stTextInput>div>div>input {
        color: #ffffff !important;
        background-color: #162032 !important; /* Cockpit Panel Blue */
        border: 1px solid #334155 !important;
    }
    .stTextInput>div>div>input:focus {
        border: 1px solid #38bdf8 !important; /* Tech Cyan Accent */
    }
    .stImage > img {
        border: 1px solid #334155;
        border-radius: 6px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    }
    hr {
        border-color: #334155;
    }
    .stSpinner > div > div {
        border-top-color: #38bdf8 !important; /* Cyan loading spinner */
    }
</style>
""", unsafe_allow_html=True)

st.title("🚀 Aerospace Multi-Modal RAG")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Ask the Manuals")
    user_query = st.chat_input("E.g., What is the structural load at Mach 0.8?")
    
    if user_query:
        st.write(f"**You:** {user_query}")
        
        with st.spinner("Searching Frankfurt Multi-Modal Index..."):
            image_filename, search_score = get_relevant_image(user_query)
            
            if not image_filename:
                st.error("No relevant documents found in the index.")
                st.stop()
                
            image_path = os.path.join("aerospace_images", image_filename)
            
        with col2:
            st.markdown(f"### Source Document: `{image_filename}`")
            st.info(f"**Relevance Score:** {search_score:.3f}")
            st.image(image_path)
            
        with st.spinner("Local M4 VLM Reading Image..."):
            
            prompt = (
                f"You are a helpful Aerospace Engineering AI. Answer the user's question: '{user_query}'\n\n"
                "Instructions:\n"
                "1. Use the provided image as your primary source of truth.\n"
                "2. If the image contains the answer, extract it and explain it clearly.\n"
                "3. If the image only provides partial context (e.g., it's a title page or a diagram), you may combine it with your general knowledge to provide a fully helpful answer.\n"
                "4. Be informative and do NOT overly restrict yourself from answering basic definitions."
            )
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            formatted_prompt = vlm_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            
            response = generate(
                vlm_model, vlm_processor, formatted_prompt, 
                image_files=[image_path], max_tokens=300, verbose=False
            )
            
            # The clean output!
            st.write(f"**AI:** {response.text}")