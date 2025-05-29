import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import ICA
from pylsl import StreamInlet, resolve_streams
from scipy.signal import coherence, welch
import time
import os
import openai

# Configure app
st.set_page_config(layout="wide", page_title="🧠 EEG Analysis Suite")
st.title("🧠 EEG Analysis Suite with GPT Interpretation")
st.markdown("Developed by **Kanwar Hamza Shuja**")

# Load OpenAI API key
openai.api_key = st.secrets.get("openai_api_key") or os.getenv("OPENAI_API_KEY")

# Helper function for AI insight

def generate_ai_insight(analysis_type, results):
    if not openai.api_key:
        return "⚠️ AI interpretation unavailable (API key not set)."
    prompt = f"""
You are a neuroscience researcher. Analyze the following {analysis_type} results from an EEG study.
Provide a detailed discussion as in the 'Discussion' section of a scientific paper, including:
- Key findings
- Neurophysiological interpretation
- Potential clinical or cognitive implications
- Recommendations for further analysis

Results:
{results}
"""
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )
        # For openai==0.28 compatibility
        content = None
        if hasattr(resp.choices[0], 'message'):
            # message is a dict-like
            content = resp.choices[0].message['content']
        else:
            content = resp.choices[0].text
        return content.strip()
    except Exception as e:
        return f"⚠️ AI interpretation failed: {e}"

# Data loading functions

def get_lsl_inlet():
    streams = resolve_streams()
    eeg_streams = [s for s in streams if hasattr(s, 'type') and s.type() == 'EEG']
    return StreamInlet(eeg_streams[0], max_chunklen=12) if eeg_streams else None

def acquire_lsl_data(inlet, duration, sfreq, n_channels):
    n_samples = int(duration * sfreq)
    data = np.zeros((n_channels, n_samples))
    idx = 0
    start = time.time()
    while idx < n_samples and (time.time() - start) < duration + 1:
        chunk, _ = inlet.pull_chunk(timeout=0.0, max_samples=50)
        if chunk:
            arr = np.array(chunk).T
            L = arr.shape[1]
            end = min(idx + L, n_samples)
            data[:, idx:end] = arr[:, :end-idx]
            idx = end
    return data[:, :idx]

def create_raw(data, sfreq, ch_names):
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(data, info)
    try:
        raw.set_montage('standard_1020', on_missing='ignore')
    except Exception:
        pass
    return raw

# EEG Analyses

def plot_time_series(raw):
    st.subheader("Time Series Plot")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(raw.times, raw.get_data().T)
    ax.set(title='EEG Time Series', xlabel='Time (s)', ylabel='Amplitude (µV)')
    st.pyplot(fig)
    stats = {
        "Mean amplitude (µV)": float(np.mean(raw.get_data())),
        "Std deviation": float(np.std(raw.get_data())),
        "Channels": len(raw.ch_names),
        "Duration (s)": float(raw.times[-1])
    }
    with st.expander("📝 AI Interpretation"):
        st.write(generate_ai_insight("Time Series", stats))

def compute_band_power(raw):
    st.subheader("Band Power Analysis")
    bands = {'Delta':(0.5,4), 'Theta':(4,8), 'Alpha':(8,13), 'Beta':(13,30), 'Gamma':(30,45)}
    selected = st.multiselect("Select channels", raw.ch_names, default=raw.ch_names)
    if not selected:
        return
    data = raw.pick(selected).get_data()
    sf = raw.info['sfreq']
    f, Pxx = welch(data[0], fs=sf, nperseg=256)
    power = {band: float(np.mean(Pxx[(f>=low)&(f<=high)])) for band, (low, high) in bands.items()}
    df = pd.DataFrame(power, index=[0]).T.rename(columns={0:'Power'})
    st.dataframe(df)
    with st.expander("📝 AI Interpretation"):
        st.write(generate_ai_insight("Band Power Analysis", df.to_dict()))

def run_ica_analysis(raw):
    st.subheader("ICA Artifact Removal (Picard)")
    n = st.slider("Components", 1, min(20, len(raw.ch_names)), 5)
    try:
        ica = ICA(n_components=n, method='picard', random_state=97)
        with st.spinner("Running ICA..."):
            ica.fit(raw)
        figs = ica.plot_components(show=False)
        for fig in figs:
            st.pyplot(fig)
        excl = st.multiselect("Exclude components", list(range(n)))
        if st.button("Apply ICA"):
            raw2 = raw.copy()
            ica.exclude = excl
            ica.apply(raw2)
            st.session_state.raw = raw2
            st.success(f"Applied ICA cleaning: excluded {excl}")
            with st.expander("📝 AI Interpretation"):
                st.write(generate_ai_insight("ICA Artifact Removal", {"excluded": excl}))
    except Exception as e:
        st.error(f"ICA failed: {e}")

def plot_topomap(raw):
    st.subheader("Topographic Map")
    # Compute PSD using Raw.compute_psd for compatibility
    psd = raw.compute_psd(fmin=1, fmax=30, verbose=False)
    psds = psd.get_data()
    freqs = psd.frequencies
    mean_psd = np.mean(psds, axis=1)
    fig, ax = plt.subplots()
    mne.viz.plot_topomap(mean_psd, raw.info, axes=ax, show=False)
    st.pyplot(fig)
    with st.expander("📝 AI Interpretation"):
        st.write(generate_ai_insight("Topographic Map", mean_psd.tolist()))
    with st.expander("📝 AI Interpretation"):
        st.write(generate_ai_insight("Topographic Map", mean_psd.tolist()))

def plot_coherence(raw):
    st.subheader("Coherence Analysis")
    ch1 = st.selectbox("Channel 1", raw.ch_names)
    ch2 = st.selectbox("Channel 2", raw.ch_names)
    if ch1 == ch2:
        st.warning("Select two different channels")
        return
    f, Cxy = coherence(raw.get_data(picks=[ch1])[0], raw.get_data(picks=[ch2])[0], fs=raw.info['sfreq'])
    fig, ax = plt.subplots()
    ax.semilogy(f, Cxy)
    ax.set(title=f'Coherence between {ch1} and {ch2}', xlabel='Frequency (Hz)', ylabel='Coherence')
    st.pyplot(fig)
    with st.expander("📝 AI Interpretation"):
        st.write(generate_ai_insight("Coherence Analysis", dict(zip(f.tolist(), Cxy.tolist()))))

# Main app

def main():
    mode = st.sidebar.radio("Data Source", ["LSL Stream", "Upload CSV(s)"])
    if mode == "LSL Stream":
        inlet = get_lsl_inlet()
        if not inlet:
            st.warning("No EEG stream found.")
            return
        dur = st.sidebar.slider("Duration (s)", 1, 60, 5)
        if st.sidebar.button("Acquire Data"):
            data = acquire_lsl_data(inlet, dur, 250, 8)
            raw = create_raw(data, 250, [f"Ch{i+1}" for i in range(8)])
            st.session_state.raw = raw
            st.success("Data acquired")
    else:
        up = st.sidebar.file_uploader("Upload CSV(s)", type="csv", accept_multiple_files=True)
        if up and st.sidebar.button("Load Data"):
            raw_list = []
            for file in up:
                df = pd.read_csv(file)
                sf = st.sidebar.number_input(f"Sampling Frequency for {file.name} (Hz)", 100, 1000, 250)
                raw = create_raw(df.values.T, sf, df.columns.tolist())
                raw_list.append(raw)
            if raw_list:
                st.session_state.raw = raw_list[0] if len(raw_list) == 1 else raw_list[0].add_channels(raw_list[1:])
                st.success("Data loaded")

    if 'raw' in st.session_state:
        raw = st.session_state.raw
        analysis = st.sidebar.selectbox("Analysis", [
            "Time Series", "Band Power", "ICA Artifact Removal", "Topographic Map", "Coherence Analysis"
        ])
        st.header(analysis)
        if analysis == "Time Series":
            plot_time_series(raw)
        elif analysis == "Band Power":
            compute_band_power(raw)
        elif analysis == "ICA Artifact Removal":
            run_ica_analysis(raw)
        elif analysis == "Topographic Map":
            plot_topomap(raw)
        elif analysis == "Coherence Analysis":
            plot_coherence(raw)

if __name__ == "__main__":
    main()
