import streamlit as st
import os
import json
from datetime import datetime
from pipeline import DiarizationPipeline
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Page config
st.set_page_config(
    page_title="Diarization Pipeline",
    page_icon="🎙️",
    layout="wide"
)

# Initialize session state
if 'logs' not in st.session_state:
    st.session_state.logs = []
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'result_text' not in st.session_state:
    st.session_state.result_text = ""

def log_callback(message):
    """Callback to capture logs in real-time for the UI"""
    st.session_state.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

# --- UI Layout ---

# Title
st.title("🎙️ Audio Diarization Pipeline")
st.markdown("Upload an audio file to process through VAD → Diarization → Transcription")

# Main content area
st.header("📤 Upload Audio")
uploaded_file = st.file_uploader(
    "Choose an audio file",
    type=['wav', 'mp3', 'm4a', 'flac'],
    help="Supported formats: WAV, MP3, M4A, FLAC"
)

if uploaded_file:
    st.audio(uploaded_file)
    
    process_button = st.button("🚀 Process Audio", type="primary", disabled=st.session_state.processing)
    
    if process_button:
        # Get config from environment variables
        hf_token = os.getenv("HF_TOKEN", "")
        whisper_endpoint = os.getenv("WHISPER_API_ENDPOINT", "")
        language = os.getenv("LANGUAGE", "ur")

        if not hf_token or not whisper_endpoint:
            st.error("❌ Critical environment variables HF_TOKEN or WHISPER_API_ENDPOINT are not set!")
        else:
            # Clear previous run's data
            st.session_state.logs = []
            st.session_state.result_text = ""
            st.session_state.processing = True
            
            # Save uploaded file to a temporary location
            temp_dir = "/app/uploads"
            output_dir = "/app/outputs/streamlit"
            os.makedirs(temp_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)
            
            input_path = os.path.join(temp_dir, f"temp_{uploaded_file.name}")
            with open(input_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Process with pipeline in a spinner
            with st.spinner("Processing... Check logs below for real-time updates"):
                try:
                    pipeline = DiarizationPipeline(
                        hf_token=hf_token,
                        whisper_endpoint=whisper_endpoint,
                        language=language,
                        log_callback=log_callback
                    )
                    
                    result = pipeline.process(input_path, output_dir)
                    
                    if result and result.get('transcriptions'):
                        st.success("✅ Processing completed!")
                        # Format results into a single string for display
                        full_transcription = ""
                        for trans in result['transcriptions']:
                            start, end, speaker, text = trans['start'], trans['end'], trans['speaker'], trans['text']
                            full_transcription += f"[{start:08.3f}s - {end:08.3f}s] {speaker}: {text}\n"
                        st.session_state.result_text = full_transcription
                        st.session_state.json_result = json.dumps(result, indent=2, ensure_ascii=False)
                    else:
                        st.error("❌ Processing failed or produced no results. Check logs below.")
                        st.session_state.result_text = "Processing failed."

                except Exception as e:
                    st.error(f"❌ A critical error occurred: {str(e)}")
                    st.session_state.result_text = f"An error occurred: {e}"
                finally:
                    st.session_state.processing = False
                    if os.path.exists(input_path):
                        os.remove(input_path)
                    st.rerun() # Rerun to update the UI elements outside the button press logic

# --- Results and Logs Display ---

st.markdown("---")
st.header("📊 Results")

# Display the formatted transcription text
if st.session_state.result_text:
    st.text_area("Formatted Transcription", value=st.session_state.result_text, height=400)
    # Add download button for the JSON result
    if 'json_result' in st.session_state:
        st.download_button(
            label="📥 Download Full JSON Results",
            data=st.session_state.json_result,
            file_name=f"transcription_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
else:
    st.info("👈 Upload an audio file and click 'Process Audio' to see results.")

st.markdown("---")
st.header("📋 Processing Logs")

# Display logs in a container
if st.session_state.logs:
    log_text = "\n".join(st.session_state.logs)
    st.text_area("Live Logs", value=log_text, height=300, key="log_area")
else:
    st.info("Logs will appear here during processing...")
