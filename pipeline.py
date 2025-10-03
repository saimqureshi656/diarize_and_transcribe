import os
import torch
import requests
import logging
from datetime import datetime
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.audio import Pipeline
from pydub import AudioSegment
from huggingface_hub import login # <-- ADDED THIS IMPORT

class DiarizationPipeline:
    def __init__(self, hf_token, whisper_endpoint, language, log_callback=None):
        self.hf_token = hf_token
        self.whisper_endpoint = whisper_endpoint
        self.language = language
        self.log_callback = log_callback
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Setup logging
        self.logger = logging.getLogger(__name__)

        # --- ADDED: Global Hugging Face login ---
        if self.hf_token:
            try:
                login(token=self.hf_token)
                self.log("✓ Successfully logged into Hugging Face Hub.")
            except Exception as e:
                self.log(f"❌ Failed to log into Hugging Face Hub: {e}", "error")
                # This error might prevent model loading, but we allow the pipeline to proceed
                # to potentially catch other issues or for cases where models might be cached.
        else:
            self.log("⚠ Hugging Face Token not provided. Model downloads might fail if authentication is required.", "warning")
        # --- END ADDITION ---
        
    def log(self, message, level="info"):
        """Log message and send to callback if available"""
        if level == "info":
            self.logger.info(message)
        elif level == "error":
            self.logger.error(message)
        elif level == "warning":
            self.logger.warning(message)
            
        if self.log_callback:
            self.log_callback(message)
    
    def detect_and_remove_beeps(self, audio_path, output_path):
        """Detect and remove beeps/rings from beginning of audio"""
        self.log("=" * 60)
        self.log("STEP 1: BEEP/RING DETECTION & REMOVAL")
        self.log("=" * 60)
        
        try:
            audio = AudioSegment.from_file(audio_path)
            
            # Simple approach: Remove first 3 seconds if they exist
            # You can enhance this with frequency analysis later
            beep_duration = min(3000, len(audio))  # 3 seconds or audio length
            
            if len(audio) > beep_duration:
                self.log(f"⏳ Analyzing first {beep_duration/1000}s for beeps/rings...")
                cleaned_audio = audio[beep_duration:]
                cleaned_audio.export(output_path, format="wav")
                self.log(f"✓ Removed {beep_duration/1000}s from beginning")
                return True, beep_duration/1000
            else:
                # Audio too short, just copy
                audio.export(output_path, format="wav")
                self.log("⚠ Audio too short, skipping beep removal")
                return True, 0
                
        except Exception as e:
            self.log(f"❌ Error during beep removal: {e}", "error")
            return False, 0
    
    def preprocess_with_vad(self, audio_path, output_path):
        """Remove non-speech parts using VAD"""
        self.log("\n" + "=" * 60)
        self.log("STEP 2: VOICE ACTIVITY DETECTION (VAD)")
        self.log("=" * 60)
        
        try:
            self.log("⏳ Loading VAD model...")
            # --- MODIFIED: Removed 'token' argument ---
            model = Model.from_pretrained("pyannote/segmentation-3.0") 
            vad_pipeline = VoiceActivityDetection(segmentation=model)
            
            HYPER_PARAMETERS = {
                "min_duration_on": 0.05,
                "min_duration_off": 0.5
            }
            vad_pipeline.instantiate(HYPER_PARAMETERS)
            vad_pipeline.to(torch.device(self.device))
            
            self.log(f"⏳ Running VAD on audio...")
            vad_result = vad_pipeline(audio_path)
            
            audio = AudioSegment.from_file(audio_path)
            speech_chunks = [
                audio[int(s.start*1000):int(s.end*1000)] 
                for s in vad_result.get_timeline().support()
            ]
            
            if not speech_chunks:
                self.log("❌ No speech detected in audio", "warning")
                return False
            
            cleaned_audio = sum(speech_chunks)
            cleaned_audio.export(output_path, format="wav")
            
            total_speech = sum([len(chunk) for chunk in speech_chunks]) / 1000
            self.log(f"✓ VAD complete! Total speech: {total_speech:.2f}s")
            return True
            
        except Exception as e:
            self.log(f"❌ Error during VAD: {e}", "error")
            return False
    
    def transcribe_chunk(self, audio_chunk_path):
        """Send audio chunk to Whisper API"""
        try:
            with open(audio_chunk_path, 'rb') as f:
                files = {'file': f}
                data = {'language': self.language}
                response = requests.post(self.whisper_endpoint, files=files, data=data)
                
                if response.status_code == 200:
                    return response.json().get("transcript", "[Transcription Error]")
                else:
                    return f"[API Error: {response.status_code}]"
        except Exception as e:
            return f"[Error: {str(e)}]"
    
    def diarize_and_transcribe(self, audio_path, output_dir):
        """Perform diarization and transcription"""
        self.log("\n" + "=" * 60)
        self.log("STEP 3: SPEAKER DIARIZATION")
        self.log("=" * 60)
        
        try:
            self.log("⏳ Loading diarization pipeline...")
            # --- MODIFIED: Removed 'token' argument ---
            diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1"
            )
            diarization_pipeline.to(torch.device(self.device))
            self.log("✓ Diarization pipeline loaded!")
            
            self.log(f"⏳ Processing diarization...")
            diarization_output = diarization_pipeline(audio_path)
            diarization = diarization_output


             # --- NEW --- Log raw diarization timestamps for testing
            self.log("\n" + "=" * 60)
            self.log("STEP 3.5: RAW DIARIZATION TIMESTAMPS")
            self.log("=" * 60)
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                self.log(f"[{turn.start:08.3f}s - {turn.end:08.3f}s] Speaker: {speaker}")
            # --- END NEW ---
            
            audio = AudioSegment.from_file(audio_path)
            
            self.log("\n" + "=" * 60)
            self.log("STEP 4: TRANSCRIPTION")
            self.log("=" * 60)
            
            results = []
            
            for i, (turn, _, speaker) in enumerate(diarization.itertracks(yield_label=True)):
                start_time = turn.start
                end_time = turn.end
                duration = end_time - start_time
                
                # Skip very short segments (likely noise)
                if duration < 0.5:
                    self.log(f"[{start_time:08.3f}s - {end_time:08.3f}s] {speaker}: [Skipped: Too short ({duration:.2f}s)]")
                    results.append({
                        "start": start_time,
                        "end": end_time,
                        "speaker": speaker,
                        "text": "[Skipped: Too short]",
                        "duration": duration
                    })
                    continue
                
                # Extract and transcribe chunk
                start_ms = int(start_time * 1000)
                end_ms = int(end_time * 1000)
                chunk = audio[start_ms:end_ms]
                
                temp_chunk_path = os.path.join(output_dir, f"temp_chunk_{i}.wav")
                chunk.export(temp_chunk_path, format="wav")
                
                self.log(f"⏳ Transcribing segment {i+1}...")
                transcription = self.transcribe_chunk(temp_chunk_path)
                
                self.log(f"[{start_time:08.3f}s - {end_time:08.3f}s] {speaker}: {transcription}")
                
                results.append({
                    "start": start_time,
                    "end": end_time,
                    "speaker": speaker,
                    "text": transcription,
                    "duration": duration
                })
                
                # Cleanup temp file
                if os.path.exists(temp_chunk_path):
                    os.remove(temp_chunk_path)
            
            self.log("\n✓ Pipeline completed successfully!")
            return results
            
        except Exception as e:
            self.log(f"❌ Error during diarization/transcription: {e}", "error")
            return None # Ensure None is returned on error
    
    def process(self, input_audio_path, output_dir):
        """Main pipeline process"""
        self.log(f"\n{'='*60}")
        self.log(f"STARTING DIARIZATION PIPELINE")
        self.log(f"Input: {input_audio_path}")
        self.log(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log(f"{'='*60}\n")
        
        # Create temp paths
        beep_removed_path = os.path.join(output_dir, "temp_beep_removed.wav")
        vad_cleaned_path = os.path.join(output_dir, "temp_vad_cleaned.wav")
        
        # Step 1: Remove beeps
        success, beep_duration = self.detect_and_remove_beeps(input_audio_path, beep_removed_path)
        if not success:
            self.log("Exiting pipeline: Beep removal failed.", "error") # <-- Added log
            return None
        
        # Step 2: VAD preprocessing
        success = self.preprocess_with_vad(beep_removed_path, vad_cleaned_path)
        if not success:
            self.log("Exiting pipeline: VAD preprocessing failed or detected no speech.", "error") # <-- Added log
            return None
        
        # Step 3 & 4: Diarization and Transcription
        results = self.diarize_and_transcribe(vad_cleaned_path, output_dir)
        
        # Cleanup temp files (only if results were successfully obtained to avoid deleting crucial debug files on failure)
        if results is not None:
            for temp_file in [beep_removed_path, vad_cleaned_path]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
        else:
            self.log("Exiting pipeline: Diarization and transcription failed.", "error") # <-- Added log
            # Optionally, keep temp files for debugging if results is None
            pass
        
        return {
            "beep_duration": beep_duration,
            "transcriptions": results
        } if results is not None else None # Ensure we return None if transcription failed
