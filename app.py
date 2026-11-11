"""VoiceCmd-DTW — Streamlit interface for isolated-word speech recognition."""

import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from voicecmd import (AudioError, DTWRecognizer, FeatureConfig, dtw, evaluate, extract_features,
                      preprocess)
from voicecmd.datasets import enroll_dir, list_labelled_files, load_labelled_dir
from voicecmd.plots import (plot_alignment, plot_confusion, plot_distances, plot_features,
                            plot_waveform)

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.join(HERE, "data", "demo")

st.set_page_config(page_title="VoiceCmd-DTW", page_icon="🎙️", layout="wide")


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


# ------------------------------------------------------------------ state
if "rec" not in st.session_state:
    st.session_state.rec = DTWRecognizer()
rec: DTWRecognizer = st.session_state.rec

st.title("🎙️ VoiceCmd-DTW")
st.caption("Isolated-word speech recognition · MFCC features · Dynamic Time Warping · built from scratch with NumPy/SciPy")

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Recogniser settings")
    full_dtw = st.checkbox("Unconstrained DTW (no band)", value=False)
    band = st.slider("Sakoe-Chiba band (fraction)", 0.05, 0.6, 0.3, 0.05, disabled=full_dtw)
    rec.band = None if full_dtw else band
    rec.k = st.slider("k nearest templates per word", 1, 5, 1)
    rec.use_vad = st.checkbox("Voice activity detection (trim silence)", value=True)
    use_thr = st.checkbox("Reject out-of-vocabulary words", value=False)
    if use_thr:
        default_thr = 4.5
        rec.reject_threshold = st.number_input("Distance threshold", 0.5, 20.0, default_thr, 0.1)
        if rec.n_templates >= 2 and st.button("Suggest threshold from templates"):
            try:
                st.info(f"Suggested threshold: {rec.suggest_threshold():.2f}")
            except RuntimeError as exc:
                st.warning(str(exc))
    else:
        rec.reject_threshold = None
    st.divider()
    st.metric("Vocabulary words", len(rec.words))
    st.metric("Templates", rec.n_templates)

tab_enrol, tab_rec, tab_eval = st.tabs(["1 · Enrol words", "2 · Recognise", "3 · Evaluate"])

# ------------------------------------------------------------------ enrol
with tab_enrol:
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Quick start")
        if st.button("Load demo vocabulary (synthetic speech)"):
            if not os.path.isdir(os.path.join(DEMO, "enroll")):
                st.error("Demo data missing. Run: python cli.py generate-demo data/demo")
            else:
                rec.clear()
                n = enroll_dir(rec, os.path.join(DEMO, "enroll"))
                st.success(f"Enrolled {n} demo templates for {len(rec.words)} words.")
        if st.button("Clear all templates"):
            rec.clear()
            st.info("All templates removed.")

        st.subheader("Add your own word")
        word = st.text_input("Word label", placeholder="e.g. hello")
        files = st.file_uploader("Upload one or more WAV recordings of this word", type=["wav"],
                                 accept_multiple_files=True, key="enrol_files")
        mic = st.audio_input("…or record it now") if hasattr(st, "audio_input") else None
        if st.button("Add to vocabulary"):
            clips = list(files or []) + ([mic] if mic is not None else [])
            if not word.strip():
                st.error("Enter a word label first.")
            elif not clips:
                st.error("Provide at least one recording.")
            else:
                added = 0
                for f in clips:
                    try:
                        y, sr = preprocess(f)
                        rec.add_template(word, y, sr)
                        added += 1
                    except AudioError as exc:
                        st.error(f"{getattr(f, 'name', 'recording')}: {exc}")
                if added:
                    st.success(f"Added {added} template(s) for '{word.strip().lower()}'.")
    with c2:
        st.subheader("Current vocabulary")
        counts = rec.template_counts()
        if counts:
            st.dataframe(pd.DataFrame({"word": list(counts), "templates": list(counts.values())}),
                         hide_index=True, width="stretch")
            buf = io.BytesIO()
            rec.save(buf)
            st.download_button("Download model (.npz)", buf.getvalue(), "voicecmd_model.npz")
        else:
            st.info("No words enrolled yet.")
        model_up = st.file_uploader("Load a saved model (.npz)", type=["npz"], key="model_up")
        if model_up is not None and st.button("Use uploaded model"):
            try:
                st.session_state.rec = DTWRecognizer.load(model_up.getvalue())
                st.rerun()
            except (ValueError, AudioError) as exc:
                st.error(str(exc))

# -------------------------------------------------------------- recognise
with tab_rec:
    if rec.n_templates == 0:
        st.info("Enrol some words first (tab 1) — the demo vocabulary takes one click.")
    else:
        st.write("Vocabulary: " + ", ".join(f"**{w}**" for w in rec.words))
        source = st.radio("Audio source", ["Upload WAV", "Record", "Demo test clip"], horizontal=True)
        audio = None
        label = None
        if source == "Upload WAV":
            audio = st.file_uploader("WAV file", type=["wav"], key="rec_upload")
        elif source == "Record":
            if hasattr(st, "audio_input"):
                audio = st.audio_input("Say one word")
            else:
                st.warning("Recording requires a newer Streamlit; use Upload instead.")
        else:
            clips = [(w, p) for w, p in list_labelled_files(os.path.join(DEMO, "test"))] if os.path.isdir(os.path.join(DEMO, "test")) else []
            clips += [(w, p) for w, p in list_labelled_files(os.path.join(DEMO, "test_noisy"))] if os.path.isdir(os.path.join(DEMO, "test_noisy")) else []
            if clips:
                names = [os.path.relpath(p, DEMO) for _, p in clips]
                choice = st.selectbox("Demo clip", names)
                path = os.path.join(DEMO, choice)
                audio = path
                label = clips[names.index(choice)][0]
            else:
                st.warning("No demo clips found.")

        if audio is not None:
            st.audio(audio)
            if st.button("Recognise", type="primary"):
                try:
                    y, sr = preprocess(audio)
                    res = rec.recognize(y, sr)
                except AudioError as exc:
                    st.error(f"Could not recognise: {exc}")
                else:
                    if res.accepted:
                        st.success(f"Recognised word: **{res.word}**")
                    else:
                        st.warning(f"Rejected as out-of-vocabulary (closest: {res.word}, distance {res.distance:.2f} above threshold).")
                    if label is not None:
                        st.caption(f"(Ground truth of this demo clip: {label})")
                    m = st.columns(5)
                    m[0].metric("DTW distance", f"{res.distance:.2f}")
                    m[1].metric("Margin", f"{res.margin:.2f}")
                    m[2].metric("Audio (s)", f"{res.audio_duration_s:.2f}")
                    m[3].metric("Processing (ms)", f"{res.elapsed_s*1000:.0f}")
                    m[4].metric("Real-time factor", f"{res.rtf:.3f}")

                    a, b = st.columns(2)
                    with a:
                        show(plot_waveform(y, sr, res.segment_s))
                        feats = rec.features_from_audio(y, sr)
                        show(plot_features(feats))
                    with b:
                        show(plot_distances(res.ranking))
                        tmpl = rec.template_features(res.best_template)
                        path = dtw(feats, tmpl, band=rec.band, return_path=True).path
                        show(plot_alignment(path, len(feats), len(tmpl)))
                    st.dataframe(pd.DataFrame(res.ranking, columns=["word", "distance"]),
                                 hide_index=True, width="stretch")
                    st.download_button("Download result (JSON)", json.dumps(res.to_dict(), indent=2),
                                       "recognition_result.json", "application/json")

# --------------------------------------------------------------- evaluate
with tab_eval:
    st.write("Score the current vocabulary on a folder of labelled clips "
             "(`root/<word>/*.wav`). The bundled demo sets contain synthetic speech from unseen pseudo-speakers.")
    sets = [d for d in ("test", "test_noisy") if os.path.isdir(os.path.join(DEMO, d))]
    if rec.n_templates == 0:
        st.info("Enrol words first.")
    elif not sets:
        st.warning("Demo test sets not found.")
    else:
        which = st.selectbox("Demo test set", sets)
        if st.button("Run evaluation"):
            data = load_labelled_dir(os.path.join(DEMO, which))
            rep = evaluate(rec, data)
            c = st.columns(3)
            c[0].metric("Accuracy", f"{rep.accuracy*100:.1f}%")
            c[1].metric("Clips", f"{rep.n_correct}/{rep.n}")
            c[2].metric("Mean RTF", f"{rep.mean_rtf:.3f}")
            show(plot_confusion(rep.confusion, rep.labels))
            st.dataframe(pd.DataFrame(rep.per_class()).T, width="stretch")
