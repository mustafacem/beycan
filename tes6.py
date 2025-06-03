from __future__ import annotations
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List

import openai
import streamlit as st
from dotenv import load_dotenv
from pydub import AudioSegment

# Load environment variables
load_dotenv()

# -----------------------------------------------------------------------------
# ▼▼▼  INITIALISE CHAT STATE & HELPER FUNCTIONS  ▼▼▼
# -----------------------------------------------------------------------------

def init_state():
    """Populate st.session_state with default values if they are missing."""
    defaults = {
        # --- COMPRESSION ---
        "compression_ratio": -20,

        # --- VOLUME ADJUSTMENTS (dB) ---
        "vocals_vol_adj": 5,
        "lead_instruments_vol_adj": 3,
        "drums_vol_adj": -2,
        "bass_vol_adj": 1,
        "homophonic_vol_adj": 0,

        # --- EQUALISATION (Hz) ---
        "vocals_low_pass": 8000,
        "vocals_high_pass": 300,
        "lead_low_pass": 7000,
        "lead_high_pass": 500,
        "drums_low_pass": 10000,
        "drums_high_pass": 50,
        "bass_low_pass": 300,
        "bass_high_pass": 20,
        "homo_low_pass": 10000,
        "homo_high_pass": 100,

        # --- PANNING (-1.0‒1.0) ---
        "vocals_pan": 0.0,
        "lead_pan": 0.0,
        "drums_pan": 0.0,
        "bass_pan": 0.0,
        "homo_pan": 0.0,

        # --- MULTI‑PASS COUNT ---
        "multi_pass_count": 1,

        # --- REVERB ---
        "reverb_delay": 100,
        "reverb_pre_delay": 50,
        "reverb_decay": 10,
        "reverb_repeats": 2,
        "reverb_mix": 30,
        "reverb_width": 0.5,

        # --- CHAT HISTORY ---
        "messages": [],
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

def update_state(changes):
    """Recursively update st.session_state with the provided key/value pairs."""
    for k, v in changes.items():
        if isinstance(v, dict):
            update_state(v)
        else:
            if k in st.session_state:
                st.session_state[k] = v

def extract_json(text):
    """Return the first JSON object found in *text* (or ``None``)."""
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None

def get_assistant_response(prompt: str) -> str:
    """Ask OpenAI to translate user *prompt* into a JSON state diff."""
    if not openai.api_key:
        return "⚠️ OPENAI_API_KEY not set. Please configure it in your environment."

    system_primer = (
        "You are an expert audio‑mixing assistant helping users adjust parameters "
        "in a Streamlit app. Respond STRICTLY with *one* JSON object whose keys are "
        "names that exist in st.session_state (see list below). The value for each "
        "key must be the new value (number or float). NEVER wrap the JSON in text, "
        "markdown, or code fences. List of adjustable keys and allowed ranges:\n"
        "- compression_ratio (int −60 to 0 dB)\n"
        "- vocals_vol_adj, lead_instruments_vol_adj, drums_vol_adj, bass_vol_adj, homophonic_vol_adj (int −10 to 10)\n"
        "- vocals_low_pass, lead_low_pass, drums_low_pass, homo_low_pass (int 1 000 to 20 000 Hz)\n"
        "- vocals_high_pass, lead_high_pass, drums_high_pass, homo_high_pass (int 20 to 1 000 Hz)\n"
        "- bass_low_pass (int 100 to 1 000 Hz), bass_high_pass (int 20 to 100 Hz)\n"
        "- vocals_pan, lead_pan, drums_pan, bass_pan, homo_pan (float −1.0 to 1.0)\n"
        "- multi_pass_count (int 1 to 5)\n"
        "- reverb_delay (int 50 to 500 ms), reverb_pre_delay (int 0 to 500 ms)\n"
        "- reverb_decay (int 5 to 20 dB/echo), reverb_repeats (int 1 to 5)\n"
        "- reverb_mix (int 0 to 100 percent), reverb_width (float 0.0 to 1.0)"
    )

    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_primer},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=200,
        )
        raw = response.choices[0].message.content.strip()
        return raw
    except Exception as e:
        return f"Error contacting OpenAI: {e}"

def chat_ui():
    """Render sidebar chat and apply any JSON deltas returned by the assistant."""
    st.sidebar.header("🎙️ AI Mixing Assistant")
    st.sidebar.caption("Ask me to tweak any mixing parameter.\n"
                      "Examples: *'Set drums pan slightly right', 'cut bass at 150 Hz', "
                      "'turn down the reverb mix to 15 %'*.")

    # Display history
    for msg in st.session_state["messages"]:
        with st.sidebar.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # New message box
    user_prompt = st.sidebar.chat_input("Ask me to adjust a parameter…")
    if user_prompt:
        st.session_state["messages"].append({"role": "user", "content": user_prompt})
        assistant_reply = get_assistant_response(user_prompt)
        st.session_state["messages"].append({"role": "assistant", "content": assistant_reply})

        # Try to parse JSON and update state – if parsing fails we keep the reply
        delta = extract_json(assistant_reply)
        if delta:
            update_state(delta)

# -----------------------------------------------------------------------------
# ▼▼▼  EXISTING AUDIO‑PROCESSING FUNCTIONS (UNCHANGED)  ▼▼▼
# -----------------------------------------------------------------------------

def save_uploadedfile(uploadedfile, path):
    """Save uploaded file to a specified path."""
    with open(os.path.join(path, uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())
    return os.path.join(path, uploadedfile.name)


def apply_compression_to_all(uploaded_files, compression_ratio):
    """Apply compression to all uploaded tracks, considering overall balance."""
    compressed_files = {}
    for category, files in uploaded_files.items():
        compressed_category_files = []
        for file_path in files:
            audio = AudioSegment.from_file(file_path, format="wav")
            compressed_audio = audio.compress_dynamic_range(threshold=compression_ratio)
            compressed_file_path = file_path.replace(".wav", "_compressed.wav")
            compressed_audio.export(compressed_file_path, format="wav")
            compressed_category_files.append(compressed_file_path)
        compressed_files[category] = compressed_category_files
    return compressed_files


def apply_volume_balancing(uploaded_files, volume_adjustments):
    balanced_files = {}
    for category, files in uploaded_files.items():
        balanced_category_files = []
        adjustment = volume_adjustments.get(category, 0)
        for file_path in files:
            audio = AudioSegment.from_file(file_path, format="wav")
            balanced_audio = audio + adjustment
            balanced_file_path = file_path.replace(".wav", "_balanced.wav")
            balanced_audio.export(balanced_file_path, format="wav")
            balanced_category_files.append(balanced_file_path)
        balanced_files[category] = balanced_category_files
    return balanced_files


def apply_equalization(uploaded_files, eq_settings, multi_pass_count=1):
    eq_files = {}
    for category, files in uploaded_files.items():
        eq_category_files = []
        low_pass = eq_settings[category]['low_pass']
        high_pass = eq_settings[category]['high_pass']
        for file_path in files:
            audio = AudioSegment.from_file(file_path, format="wav")
            eq_audio = audio
            for _ in range(multi_pass_count):
                eq_audio = eq_audio.low_pass_filter(low_pass).high_pass_filter(high_pass)
            eq_file_path = file_path.replace(".wav", "_eq.wav")
            eq_audio.export(eq_file_path, format="wav")
            eq_category_files.append(eq_file_path)
        eq_files[category] = eq_category_files
    return eq_files


def apply_panning(uploaded_files, panning_settings):
    panned_files = {}
    for category, files in uploaded_files.items():
        panned_category_files = []
        pan_value = panning_settings.get(category, 0)
        for file_path in files:
            audio = AudioSegment.from_file(file_path, format="wav")
            panned_audio = audio.pan(pan_value)
            panned_file_path = file_path.replace(".wav", "_panned.wav")
            panned_audio.export(panned_file_path, format="wav")
            panned_category_files.append(panned_file_path)
        panned_files[category] = panned_category_files
    return panned_files


def apply_reverb(uploaded_files, reverb_settings):
    reverb_files = {}
    delay = reverb_settings['delay']
    pre_delay = reverb_settings.get('pre_delay', 0)
    decay = reverb_settings['decay']
    repeats = reverb_settings['repeats']
    reverb_mix = reverb_settings['mix']
    reverb_width = reverb_settings.get('width', 0.0)

    for category, files in uploaded_files.items():
        reverb_category_files = []
        for file_path in files:
            audio = AudioSegment.from_file(file_path, format="wav")
            wet_duration = len(audio) + pre_delay + delay * repeats
            wet_audio = AudioSegment.silent(duration=wet_duration)
            for i in range(1, repeats + 1):
                echo = audio.apply_gain(-decay * i)
                total_delay = pre_delay + delay * i
                if reverb_width > 0:
                    echo_left = echo.pan(-reverb_width)
                    echo_right = echo.pan(reverb_width)
                    delayed_left = AudioSegment.silent(duration=total_delay) + echo_left
                    delayed_right = AudioSegment.silent(duration=total_delay) + echo_right
                    wet_audio = wet_audio.overlay(delayed_left).overlay(delayed_right)
                else:
                    delayed = AudioSegment.silent(duration=total_delay) + echo
                    wet_audio = wet_audio.overlay(delayed)
            wet_audio = wet_audio.low_pass_filter(8000)

            if reverb_mix == 0:
                final_audio = audio
            elif reverb_mix == 100:
                final_audio = wet_audio
            else:
                dry_scale = (100 - reverb_mix) / 100.0
                wet_scale = reverb_mix / 100.0
                dry_gain = 20 * math.log10(dry_scale) if dry_scale > 0 else -120
                wet_gain = 20 * math.log10(wet_scale) if wet_scale > 0 else -120
                dry_audio = audio.apply_gain(dry_gain)
                wet_audio_scaled = wet_audio.apply_gain(wet_gain)
                final_audio = dry_audio.overlay(wet_audio_scaled)

            reverb_file_path = file_path.replace(".wav", "_reverb.wav")
            final_audio.export(reverb_file_path, format="wav")
            reverb_category_files.append(reverb_file_path)
        reverb_files[category] = reverb_category_files
    return reverb_files


def combine_processed_tracks(processed_files, output_filename="final_mix.wav"):
    final_mix = None
    for category, files in processed_files.items():
        for file_path in files:
            track = AudioSegment.from_file(file_path, format="wav")
            if final_mix is None:
                final_mix = track
            else:
                final_mix = final_mix.overlay(track)
    if final_mix:
        final_mix.export(output_filename, format="wav")
        return output_filename
    return None

# -----------------------------------------------------------------------------
# ▼▼▼  MAIN STREAMLIT APP  ▼▼▼
# -----------------------------------------------------------------------------

def main():
    openai.api_key = os.getenv("OPENAI_API_KEY", "")
    
    # Initialize state and chat UI
    init_state()
    chat_ui()  # Draw the chat sidebar first so it can influence the UI below

    st.title("KUYAM.io - Track Input Management")

    st.markdown(
        """
        **Welcome to KUYAM.io!** Upload your multi-track WAV files for automated mixing and mastering.

        ### Instructions:
        1. Upload the following tracks in WAV format:
            - **Vocals**: Lead and backing vocal tracks.
            - **Homophonic Instruments**: Rhythm guitars (up to 2), keyboards/piano (up to 2).
            - **Low-Frequency Instruments**: Bass.
            - **Drums/Rhythmic Tracks**: Kick, snare, cymbals.
            - **Lead Instruments**: Lead guitar or synth lead.
        2. Adjust the sliders to modify compression, volume, EQ, panning, reverb, and multi-pass filter settings.
        3. You can also ask the AI assistant (left sidebar) to tweak these parameters via chat.
        """
    )

    # ------------------------------------------------------------------
    # COMPRESSION
    # ------------------------------------------------------------------
    st.subheader("🔧 Compression")
    st.slider(
        "Set compression threshold (dB):",
        min_value=-60,
        max_value=0,
        value=st.session_state["compression_ratio"],
        key="compression_ratio",
    )

    # ------------------------------------------------------------------
    # VOLUME BALANCING
    # ------------------------------------------------------------------
    st.subheader("🎚️ Volume Balancing")
    st.slider("Vocals Volume Adjustment (dB):", -10, 10, st.session_state["vocals_vol_adj"], key="vocals_vol_adj")
    st.slider("Lead Instruments Volume Adjustment (dB):", -10, 10, st.session_state["lead_instruments_vol_adj"], key="lead_instruments_vol_adj")
    st.slider("Drums Volume Adjustment (dB):", -10, 10, st.session_state["drums_vol_adj"], key="drums_vol_adj")
    st.slider("Bass Volume Adjustment (dB):", -10, 10, st.session_state["bass_vol_adj"], key="bass_vol_adj")
    st.slider("Homophonic Instruments Volume Adjustment (dB):", -10, 10, st.session_state["homophonic_vol_adj"], key="homophonic_vol_adj")

    # ------------------------------------------------------------------
    # EQUALISATION
    # ------------------------------------------------------------------
    st.subheader("🎛️ Equalisation")
    st.slider("Vocals Low Pass Filter (Hz):", 1000, 20000, st.session_state["vocals_low_pass"], key="vocals_low_pass")
    st.slider("Vocals High Pass Filter (Hz):", 20, 1000, st.session_state["vocals_high_pass"], key="vocals_high_pass")

    st.slider("Lead Instruments Low Pass Filter (Hz):", 1000, 20000, st.session_state["lead_low_pass"], key="lead_low_pass")
    st.slider("Lead Instruments High Pass Filter (Hz):", 20, 1000, st.session_state["lead_high_pass"], key="lead_high_pass")

    st.slider("Drums Low Pass Filter (Hz):", 1000, 20000, st.session_state["drums_low_pass"], key="drums_low_pass")
    st.slider("Drums High Pass Filter (Hz):", 20, 1000, st.session_state["drums_high_pass"], key="drums_high_pass")

    st.slider("Bass Low Pass Filter (Hz):", 100, 1000, st.session_state["bass_low_pass"], key="bass_low_pass")
    st.slider("Bass High Pass Filter (Hz):", 20, 100, st.session_state["bass_high_pass"], key="bass_high_pass")

    st.slider("Homophonic Low Pass Filter (Hz):", 1000, 20000, st.session_state["homo_low_pass"], key="homo_low_pass")
    st.slider("Homophonic High Pass Filter (Hz):", 20, 1000, st.session_state["homo_high_pass"], key="homo_high_pass")

    # ------------------------------------------------------------------
    # PANNING
    # ------------------------------------------------------------------
    st.subheader("🧭 Panning")
    st.slider("Vocals Panning (-1: Left, 0: Center, 1: Right):", -1.0, 1.0, st.session_state["vocals_pan"], key="vocals_pan")
    st.slider("Lead Instruments Panning (-1: Left, 0: Center, 1: Right):", -1.0, 1.0, st.session_state["lead_pan"], key="lead_pan")
    st.slider("Drums Panning (-1: Left, 0: Center, 1: Right):", -1.0, 1.0, st.session_state["drums_pan"], key="drums_pan")
    st.slider("Bass Panning (-1: Left, 0: Center, 1: Right):", -1.0, 1.0, st.session_state["bass_pan"], key="bass_pan")
    st.slider("Homophonic Panning (-1: Left, 0: Center, 1: Right):", -1.0, 1.0, st.session_state["homo_pan"], key="homo_pan")

    # ------------------------------------------------------------------
    # MULTI‑PASS
    # ------------------------------------------------------------------
    st.subheader("🔁 Multi‑Pass EQ")
    st.slider(
        "Multi-Pass Filter Count for Equalization:",
        min_value=1,
        max_value=5,
        value=st.session_state["multi_pass_count"],
        key="multi_pass_count",
        help="Number of times the low-pass and high-pass filters are applied.",
    )

    # ------------------------------------------------------------------
    # REVERB
    # ------------------------------------------------------------------
    st.subheader("🌊 Reverb")
    st.slider("Reverb Base Delay (ms):", 50, 500, st.session_state["reverb_delay"], key="reverb_delay")
    st.slider("Reverb Pre-Delay (ms):", 0, 500, st.session_state["reverb_pre_delay"], key="reverb_pre_delay")
    st.slider("Reverb Decay (dB per echo):", 5, 20, st.session_state["reverb_decay"], key="reverb_decay")
    st.slider("Reverb Repeats:", 1, 5, st.session_state["reverb_repeats"], key="reverb_repeats")
    st.slider("Reverb Wet/Dry Mix (%):", 0, 100, st.session_state["reverb_mix"], key="reverb_mix")
    st.slider("Reverb Width (Stereo Spread):", 0.0, 1.0, st.session_state["reverb_width"], key="reverb_width")

    # ------------------------------------------------------------------
    # CALCULATED DICTIONARIES (use latest state)
    # ------------------------------------------------------------------
    compression_ratio = st.session_state["compression_ratio"]

    volume_adjustments = {
        "Vocals": st.session_state["vocals_vol_adj"],
        "Lead Instruments": st.session_state["lead_instruments_vol_adj"],
        "Drums/Rhythmic Tracks": st.session_state["drums_vol_adj"],
        "Low-Frequency Instruments": st.session_state["bass_vol_adj"],
        "Homophonic Instruments": st.session_state["homophonic_vol_adj"],
    }

    eq_settings = {
        "Vocals": {
            'low_pass': st.session_state["vocals_low_pass"],
            'high_pass': st.session_state["vocals_high_pass"],
        },
        "Lead Instruments": {
            'low_pass': st.session_state["lead_low_pass"],
            'high_pass': st.session_state["lead_high_pass"],
        },
        "Drums/Rhythmic Tracks": {
            'low_pass': st.session_state["drums_low_pass"],
            'high_pass': st.session_state["drums_high_pass"],
        },
        "Low-Frequency Instruments": {
            'low_pass': st.session_state["bass_low_pass"],
            'high_pass': st.session_state["bass_high_pass"],
        },
        "Homophonic Instruments": {
            'low_pass': st.session_state["homo_low_pass"],
            'high_pass': st.session_state["homo_high_pass"],
        },
    }

    panning_settings = {
        "Vocals": st.session_state["vocals_pan"],
        "Lead Instruments": st.session_state["lead_pan"],
        "Drums/Rhythmic Tracks": st.session_state["drums_pan"],
        "Low-Frequency Instruments": st.session_state["bass_pan"],
        "Homophonic Instruments": st.session_state["homo_pan"],
    }

    multi_pass_count = st.session_state["multi_pass_count"]

    reverb_settings = {
        'delay': st.session_state["reverb_delay"],
        'pre_delay': st.session_state["reverb_pre_delay"],
        'decay': st.session_state["reverb_decay"],
        'repeats': st.session_state["reverb_repeats"],
        'mix': st.session_state["reverb_mix"],
        'width': st.session_state["reverb_width"],
    }

    # ------------------------------------------------------------------
    # FILE UPLOADS & PROCESSING BUTTONS (UNCHANGED)                           
    # ------------------------------------------------------------------
    track_categories = {
        "Vocals": ["vocals_lead", "vocals_back"],
        "Homophonic Instruments": ["rhythm_guitar_1", "rhythm_guitar_2", "keyboard_1", "keyboard_2"],
        "Low-Frequency Instruments": ["bass"],
        "Drums/Rhythmic Tracks": ["kick", "snare", "cymbals"],
        "Lead Instruments": ["lead_guitar_or_synth"],
    }

    uploaded_files = {}
    upload_path = "uploads"
    os.makedirs(upload_path, exist_ok=True)

    for category, file_tags in track_categories.items():
        st.subheader(category)
        category_files = []
        for tag in file_tags:
            uploaded_file = st.file_uploader(f"Upload {tag}.wav", type=["wav"], key=tag)
            if uploaded_file is not None:
                filepath = save_uploadedfile(uploaded_file, upload_path)
                category_files.append(filepath)
        if category_files:
            uploaded_files[category] = category_files

    # ------------------------------------------------------------------
    # BUTTONS (identical logic, untouched)                                  
    # ------------------------------------------------------------------

    if st.button("Apply Compression"):
        if uploaded_files:
            compressed_files = apply_compression_to_all(uploaded_files, compression_ratio)
            st.write("### Compressed Tracks:")
            for category, files in compressed_files.items():
                st.write(f"**{category}:**")
                for file in files:
                    with open(file, "rb") as f:
                        st.download_button(label=f"Download {os.path.basename(file)}", data=f, file_name=os.path.basename(file))
        else:
            st.warning("No files uploaded yet.")

    if st.button("Apply Volume Balancing"):
        if uploaded_files:
            compressed_files = apply_compression_to_all(uploaded_files, compression_ratio)
            balanced_files = apply_volume_balancing(compressed_files, volume_adjustments)
            st.write("### Balanced Tracks:")
            for category, files in balanced_files.items():
                st.write(f"**{category}:**")
                for file in files:
                    with open(file, "rb") as f:
                        st.download_button(label=f"Download {os.path.basename(file)}", data=f, file_name=os.path.basename(file))
        else:
            st.warning("No files uploaded yet.")

    if st.button("Apply Equalization"):
        if uploaded_files:
            compressed_files = apply_compression_to_all(uploaded_files, compression_ratio)
            balanced_files = apply_volume_balancing(compressed_files, volume_adjustments)
            eq_files = apply_equalization(balanced_files, eq_settings, multi_pass_count)
            st.write("### Equalized Tracks:")
            for category, files in eq_files.items():
                st.write(f"**{category}:**")
                for file in files:
                    with open(file, "rb") as f:
                        st.download_button(label=f"Download {os.path.basename(file)}", data=f, file_name=os.path.basename(file))
        else:
            st.warning("No files uploaded yet.")

    if st.button("Apply Panning"):
        if uploaded_files:
            compressed_files = apply_compression_to_all(uploaded_files, compression_ratio)
            balanced_files = apply_volume_balancing(compressed_files, volume_adjustments)
            eq_files = apply_equalization(balanced_files, eq_settings, multi_pass_count)
            panned_files = apply_panning(eq_files, panning_settings)
            st.write("### Panned Tracks:")
            for category, files in panned_files.items():
                st.write(f"**{category}:**")
                for file in files:
                    with open(file, "rb") as f:
                        st.download_button(label=f"Download {os.path.basename(file)}", data=f, file_name=os.path.basename(file))
        else:
            st.warning("No files uploaded yet.")

    if st.button("Apply Reverb"):
        if uploaded_files:
            compressed_files = apply_compression_to_all(uploaded_files, compression_ratio)
            balanced_files = apply_volume_balancing(compressed_files, volume_adjustments)
            eq_files = apply_equalization(balanced_files, eq_settings, multi_pass_count)
            panned_files = apply_panning(eq_files, panning_settings)
            reverb_files = apply_reverb(panned_files, reverb_settings)
            st.write("### Reverb Tracks:")
            for category, files in reverb_files.items():
                st.write(f"**{category}:**")
                for file in files:
                    with open(file, "rb") as f:
                        st.download_button(label=f"Download {os.path.basename(file)}", data=f, file_name=os.path.basename(file))
        else:
            st.warning("No files uploaded yet.")

    if st.button("Process All Steps"):
        if uploaded_files:
            compressed_files = apply_compression_to_all(uploaded_files, compression_ratio)
            balanced_files = apply_volume_balancing(compressed_files, volume_adjustments)
            eq_files = apply_equalization(balanced_files, eq_settings, multi_pass_count)
            panned_files = apply_panning(eq_files, panning_settings)
            reverb_files = apply_reverb(panned_files, reverb_settings)

            st.write("### Processed Tracks (All Steps Applied):")
            for category, files in reverb_files.items():
                st.write(f"**{category}:**")
                for file in files:
                    with open(file, "rb") as f:
                        st.download_button(label=f"Download {os.path.basename(file)}", data=f, file_name=os.path.basename(file))

            final_mix_path = combine_processed_tracks(reverb_files, output_filename="final_mix.wav")
            if final_mix_path and os.path.exists(final_mix_path):
                with open(final_mix_path, "rb") as f:
                    st.download_button(label="Download Final Mixed Track", data=f, file_name=os.path.basename(final_mix_path))
        else:
            st.warning("No files uploaded yet.")

    st.markdown("---")
    st.markdown("Once all files are uploaded and processed, your final unified WAV can be downloaded above.")

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
