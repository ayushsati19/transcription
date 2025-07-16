import whisper
import gradio as gr
import os # Import os for file path handling

# --- Configuration ---
# You can choose a different model size if 'mini' is too small or too large.
# Common options: "tiny", "base", "small", "medium", "large"
# For better accuracy, "base" or "small" are often good starting points.
# "mini" is very fast but might have lower accuracy.
WHISPER_MODEL_SIZE = "tiny" 

# --- Load Whisper Model ---
# This will download the model the first time it's run.
# Set fp16=False if you are on a CPU or don't have a compatible GPU for float16 inference.
# For 'mini' model, fp16=False is often suitable for CPU.
print(f"Loading Whisper model: {WHISPER_MODEL_SIZE}...")
try:
    model = whisper.load_model(WHISPER_MODEL_SIZE, device="cpu") # Force CPU for broader compatibility
    print(f"Whisper model '{WHISPER_MODEL_SIZE}' loaded successfully on CPU.")
except Exception as e:
    print(f"Error loading Whisper model: {e}")
    print("Please ensure you have an internet connection for download (if not cached) and sufficient memory.")
    # Exit or handle error gracefully if model loading fails
    exit()

# --- Transcription Function ---
def transcribe(audio_filepath):
    """
    Transcribes an audio file using the loaded Whisper model.

    Args:
        audio_filepath (str): The file path to the audio to be transcribed.
                              Gradio's microphone input provides a filepath.

    Returns:
        str: The transcribed text.
    """
    if audio_filepath is None:
        return "Please provide audio input."

    # Load audio and pad/trim it to fit 30 seconds
    # Whisper's load_audio function handles resampling to 16kHz automatically.
    try:
        audio = whisper.load_audio(audio_filepath)
        audio = whisper.pad_or_trim(audio)
    except Exception as e:
        return f"Error processing audio file: {e}"

    # Make log-Mel spectrogram and move to the same device as the model
    # The model is loaded on "cpu" so we move mel to "cpu"
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # Detect the spoken language
    # This returns a tuple: (language_token_id, probability_dictionary)
    _, probs = model.detect_language(mel)
    detected_language = max(probs, key=probs.get)
    print(f"Detected language: {detected_language} with probability {probs[detected_language]:.4f}")

    # Decode the audio
    # Set fp16=False for decoding if the model was loaded with fp16=False or on CPU
    options = whisper.DecodingOptions(fp16=False, language=detected_language) # Pass detected language for better accuracy
    
    try:
        result = whisper.decode(model, mel, options)
    except Exception as e:
        return f"Error during decoding: {e}"
        
    return result.text

# --- Gradio Interface Setup ---
print("\nLaunching Gradio Web UI...")
try:
    gr.Interface(
        title='OpenAI Whisper ASR Gradio Web UI',
        description='Speak into your microphone and get real-time transcription using the Whisper ASR model.',
        fn=transcribe,
        inputs=[
            # CORRECTED: Use 'sources=["microphone"]' for modern Gradio versions
            gr.Audio(sources=["microphone"], type="filepath", label="Speak Here")
        ],
        outputs=[
            # Textbox output for the transcription
            gr.Textbox(label="Transcription", lines=3)
        ],
        live=True, # Enable live updates as you speak
        # Optional: Add an example if you have a pre-recorded audio file for testing
        # examples=[["path/to/your/example_audio.wav"]]
    ).launch()
except Exception as e:
    print(f"Error launching Gradio interface: {e}")
    print("Please ensure Gradio is installed (`pip install gradio`) and your port is available.")

