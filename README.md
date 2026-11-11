# VoiceCmd-DTW — Isolated-Word Speech Recognition (MFCC + Dynamic Time Warping)

**Student:** HIRTHIK P · **Register No.:** RA2311003050442 · **Subject:** Speech Recognition

A small-vocabulary, template-based speech recogniser in which every stage of the
classical pipeline is implemented from first principles with NumPy/SciPy (no speech
toolkits, no pretrained models, no internet or FFmpeg needed):

```
WAV -> resample 16 kHz -> validate/normalise -> energy VAD -> MFCC(12)+logE +delta (26-D, CMVN)
    -> DTW against every template (Sakoe-Chiba band) -> k-NN per word -> word / reject
```

## Features
- WAV input (8/16/32-bit PCM and float, mono or stereo, any sample rate) with validation and clear errors
- MFCC from scratch: pre-emphasis, Hamming framing, FFT, mel filterbank, DCT, liftering, deltas, CMVN
- Adaptive energy-based voice activity detection (end-point detection)
- DTW with Sakoe-Chiba band and symmetric step weights; k-nearest-template scoring
- Optional out-of-vocabulary rejection with a threshold calibrated from your templates
- Streamlit app: enrol words (upload or microphone), recognise, waveform / MFCC / DTW-path plots,
  timing and real-time factor, evaluation with confusion matrix, model save/load, JSON export
- Command-line tool, 42 unit tests, scripted reproducible evaluation, PDF report generator

## Setup
Python 3.10+ is required.
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run the web app
```bash
streamlit run app.py
```
1. **Enrol** tab -> *Load demo vocabulary* (or add your own words with WAV files / microphone;
   record 3-5 clear examples per word).
2. **Recognise** tab -> upload / record / pick a demo clip -> *Recognise*.
3. **Evaluate** tab -> score the vocabulary on the bundled demo test sets.

Microphone recording needs a recent Streamlit and browser microphone permission; file upload always works.

## Command line
```bash
python cli.py generate-demo data/demo                       # (re)create synthetic demo clips
python cli.py enroll data/demo/enroll --out model.npz       # folder layout: root/<word>/*.wav
python cli.py recognize model.npz data/demo/test/go/spk20_rep0.wav
python cli.py evaluate model.npz data/demo/test
```
To use real speech, record WAV files into `myvocab/<word>/*.wav` and enrol that folder.

## Tests and reproducing the results
```bash
python -m pytest -q                         # 42 tests
python scripts/evaluate_synthetic.py        # ~2 min; writes results/results.json + confusion_clean.png
python report/build_report.py               # builds report/VoiceCmd-DTW_Submission_Report.pdf
python report/build_report.py --github <repo-url> --video <demo-video-url>   # adds clickable links
```

## Results (synthetic speech, speaker-independent)
Templates: 5 speakers x 2 clean repetitions per word. Tests: 180 utterances from 10 unseen
pseudo-speakers (6 words). White noise added to test clips only.

| SNR | clean | 30 dB | 20 dB | 15 dB | 10 dB | 5 dB | 0 dB |
|---|---|---|---|---|---|---|---|
| Accuracy | 100.0% | 100.0% | 99.4% | 92.8% | 87.2% | 72.2% | 42.8% |

Ablations at 10 dB SNR: no CMVN 21.1%, no delta features 85.0%, no VAD 96.7% (the energy VAD is the
weakest component under noise). Processing runs at roughly 0.03x real time on a laptop-class CPU.
Full tables: `results/results.json`.

**Important:** these numbers come from *synthetic formant speech* (`voicecmd/synthetic.py`), which exists so the
whole system can be exercised and measured without a corpus. They validate the implementation and show the
effect of design choices; they are **not** a claim about accuracy on natural human speech.

## Project layout
```
app.py                    Streamlit interface
cli.py                    command-line interface
voicecmd/audio_io.py      load / validate / resample / normalise
voicecmd/features.py      MFCC, deltas, CMVN
voicecmd/vad.py           energy-based end-point detection
voicecmd/dtw.py           banded DTW (+ warping path)
voicecmd/recognizer.py    templates, k-NN scoring, rejection, save/load
voicecmd/evaluation.py    accuracy, confusion matrix, per-class metrics
voicecmd/synthetic.py     formant speech generator (demo/tests/evaluation)
voicecmd/plots.py         figures
scripts/evaluate_synthetic.py   reproducible evaluation
report/build_report.py    PDF report generator
tests/                    pytest suite
data/demo/                synthetic demo clips (enroll/, test/, test_noisy/)
```

## Limitations
Isolated words only (one word per clip), vocabulary limited to enrolled words, speaker-dependent
(enrol the speaker's own voice for best results), WAV input only, and the energy VAD degrades in strong noise.
