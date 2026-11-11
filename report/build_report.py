"""Build the academic submission PDF from the real evaluation results.

Usage:
    python report/build_report.py [--github URL] [--video URL] [--docs URL] [--out PATH]

Run ``python scripts/evaluate_synthetic.py`` first so results/results.json exists.
Optional URLs become clickable links in the Submission Details table.
"""

import argparse
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

STUDENT = "HIRTHIK P"
REG_NO = "RA2311003050442"
TITLE = "VoiceCmd-DTW"
SUBTITLE = "Isolated-Word Speech Recognition using MFCC Features and Dynamic Time Warping"

NAVY = colors.HexColor("#1f3a5f")
LIGHT = colors.HexColor("#eef3f9")
GRID = colors.HexColor("#b8c4d4")


def count_tests() -> int:
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
                          cwd=ROOT, capture_output=True, text=True)
    m = re.search(r"(\d+) passed", proc.stdout)
    if proc.returncode != 0 or not m or "failed" in proc.stdout.splitlines()[-1]:
        raise SystemExit("Unit tests are not all passing; refusing to build the report.\n" + proc.stdout[-800:])
    return int(m.group(1))


def pipeline_figure(path):
    steps = ["WAV input\n(any rate,\nmono/stereo)", "Pre-process\nresample 16 kHz\nvalidate, normalise", "Energy VAD\nend-point\ndetection",
             "MFCC + log-E\n+ delta, CMVN\n(26-D, 10 ms)", "DTW vs. every\ntemplate\n(Sakoe-Chiba band)", "k-NN per word\nreject / accept\n-> word"]
    fig, ax = plt.subplots(figsize=(10, 1.9))
    ax.set_xlim(0, 10.7)
    ax.set_ylim(0, 1.9)
    ax.axis("off")
    w, gap = 1.5, 0.28
    for i, s in enumerate(steps):
        x = 0.05 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 0.3), w, 1.3, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc="#eef3f9", ec="#1f3a5f", lw=1.2))
        ax.text(x + w / 2, 0.95, s, ha="center", va="center", fontsize=8.2, color="#1f3a5f")
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + w + gap - 0.02, 0.95), xytext=(x + w + 0.02, 0.95),
                        arrowprops=dict(arrowstyle="-|>", color="#1f3a5f", lw=1.3))
    fig.savefig(path, dpi=200, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--github")
    ap.add_argument("--video")
    ap.add_argument("--docs")
    ap.add_argument("--out", default=os.path.join(ROOT, "report", "VoiceCmd-DTW_Submission_Report.pdf"))
    args = ap.parse_args()

    with open(os.path.join(ROOT, "results", "results.json")) as f:
        res = json.load(f)
    n_tests = count_tests()
    pipe_png = os.path.join(ROOT, "results", "pipeline.png")
    pipeline_figure(pipe_png)

    ss = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=ss["BodyText"], fontName="Helvetica", fontSize=9.6, leading=13.6)
    small = ParagraphStyle("small", parent=body, fontSize=8.2, leading=11, textColor=colors.HexColor("#444444"))
    cell = ParagraphStyle("cell", parent=body, fontSize=9, leading=12)
    cellb = ParagraphStyle("cellb", parent=cell, fontName="Helvetica-Bold")
    head = ParagraphStyle("head", parent=cell, fontName="Helvetica-Bold", textColor=colors.white)
    h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=15, textColor=NAVY,
                        spaceBefore=9, spaceAfter=4)
    title = ParagraphStyle("title", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=22, textColor=NAVY,
                           alignment=TA_CENTER, spaceAfter=2, leading=26)
    sub = ParagraphStyle("sub", parent=body, alignment=TA_CENTER, fontSize=10.5, textColor=colors.HexColor("#333333"))
    tag = ParagraphStyle("tag", parent=body, alignment=TA_CENTER, fontName="Helvetica-Bold", fontSize=9.5,
                         textColor=colors.HexColor("#5a6b82"), spaceBefore=4, spaceAfter=6)

    def link(text, url):
        return Paragraph(f'<link href="{url}" color="#1a5fb4"><u>{text}</u></link>', cell)

    def table(rows, widths, header=True, align_right_from=None):
        data = []
        for ri, r in enumerate(rows):
            line = []
            for c in r:
                if isinstance(c, str):
                    c = Paragraph(c, head if (header and ri == 0) else cell)
                line.append(c)
            data.append(line)
        t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
        st = [("GRID", (0, 0), (-1, -1), 0.5, GRID), ("VALIGN", (0, 0), (-1, -1), "TOP"),
              ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
              ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6)]
        if header:
            st += [("BACKGROUND", (0, 0), (-1, 0), NAVY)]
            st += [("BACKGROUND", (0, i), (-1, i), LIGHT) for i in range(2, len(rows), 2)]
        t.setStyle(TableStyle(st))
        return t

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#5a6b82"))
        canvas.drawString(18 * mm, 10 * mm, f"{TITLE} - Academic Project Submission")
        canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {doc.page}")
        canvas.restoreState()

    W = A4[0] - 36 * mm
    story = []
    story += [Paragraph(TITLE, title), Paragraph(SUBTITLE, sub),
              Paragraph("PROJECT SUBMISSION REPORT - SPEECH RECOGNITION", tag)]

    story += [Paragraph("Project Overview", h1), Paragraph(
        "This project presents the design and implementation of <b>VoiceCmd-DTW</b>, a speaker-dependent, "
        "template-based isolated-word speech recognition system. Unlike deep-learning ASR services, every stage of the "
        "classical pipeline is implemented from first principles with NumPy and SciPy: MFCC feature extraction, delta features, "
        "cepstral mean/variance normalisation, energy-based end-point detection, and Dynamic Time Warping with a Sakoe-Chiba band. "
        "A Streamlit web application lets a user enrol a vocabulary from uploaded or recorded WAV audio, recognise new utterances, "
        "inspect the signal, features and DTW alignment, and evaluate accuracy with a confusion matrix.", body)]

    rows = [["Field", "Information"],
            ["Subject", "Speech Recognition"],
            ["Topic", "Implementation of an Isolated-Word Speech Recognition System using MFCC and Dynamic Time Warping"],
            ["Source Code", link("GitHub repository", args.github) if args.github else
             "Project folder <b>voicecmd-dtw/</b>, submitted together with this report"],
            ["Documentation", link("README and setup instructions", args.docs or args.github) if (args.docs or args.github) else
             "<b>README.md</b> in the project folder (installation, usage, CLI, architecture)"],
            ["Demonstration", link("Project demo video", args.video) if args.video else
             "Run <b>streamlit run app.py</b>; one-click demo vocabulary and demo test clips are included"]]
    story += [Paragraph("Submission Details", h1), table(rows, [38 * mm, W - 38 * mm])]

    story += [Paragraph("Student Details", h1),
              table([["S. No.", "Student Name", "Registration Number"], ["1", STUDENT, REG_NO]],
                    [18 * mm, (W - 18 * mm) / 2, (W - 18 * mm) / 2])]

    feats = [["Component", "Description"],
             ["Audio Input and Validation", "Loads 8/16/32-bit PCM and float WAV files (mono or stereo), resamples to 16 kHz, removes DC offset, "
              "normalises level, and rejects empty, corrupt, too-short, too-long or silent clips. Microphone recording is supported in the app."],
             ["Feature Extraction", "From-scratch MFCC: 25 ms Hamming frames, 10 ms hop, 26-filter mel bank, 12 liftered cepstra + log energy, "
              "first-order deltas (26 dimensions) and per-utterance CMVN."],
             ["End-Point Detection", "Adaptive energy-threshold voice activity detection that trims leading/trailing silence, ignores clicks and keeps closures inside a word."],
             ["DTW Recognition", "Symmetric-step DTW with a Sakoe-Chiba band, row-vectorised via a cumulative-minimum identity; k-nearest-template scoring per word "
              "and an optional rejection threshold calibrated by leave-one-out for out-of-vocabulary words."],
             ["Results and Performance", "Predicted word, DTW distance, confidence margin, waveform, MFCC heat-map and warping path; audio duration, processing time and "
              "real-time factor; JSON export, model save/load (.npz), accuracy and confusion matrix for evaluation sets."],
             ["Testing and Reproducibility", f"{n_tests} automated tests (pytest) cover audio I/O, features, DTW, VAD, recognition, persistence and a headless UI flow; "
              "evaluation is fully scripted and seeded (scripts/evaluate_synthetic.py)."]]
    story += [Paragraph("Key Features and Deliverables", h1), table(feats, [42 * mm, W - 42 * mm])]
    story += [Spacer(1, 6), Paragraph("<b>Technology Stack:</b> Python · NumPy · SciPy · Streamlit · Matplotlib · pytest", body)]

    # ----------------------------------------------------------- page 2: method + results
    from reportlab.platypus import PageBreak
    story += [PageBreak(), Paragraph("System Architecture", h1),
              Image(pipe_png, width=W, height=W * 0.205),
              Spacer(1, 4),
              Paragraph("The recogniser stores one MFCC sequence per enrolled recording (a <i>template</i>). An unknown utterance is converted to the same "
                        "representation and aligned to every template with DTW, which absorbs differences in speaking rate. The distance is the accumulated "
                        "path cost normalised by the sum of both sequence lengths; the vocabulary word whose best template(s) give the smallest distance is returned.", body)]

    s = res["setup"]
    story += [Paragraph("Evaluation Protocol", h1), Paragraph(
        f"Because no recorded corpus is bundled, the system is evaluated on <b>synthetic formant speech</b> produced by the included generator "
        f"(6 command words - go, stop, left, right, up, down - rendered by pseudo-speakers that differ in pitch, vocal-tract length and speaking rate). "
        f"Templates: {len(s['train_speakers'])} speakers x 2 clean repetitions ({s['templates_per_word']} per word). Tests: <b>{s['test_utterances']} utterances from "
        f"{len(s['test_speakers'])} unseen speakers</b>, so the evaluation is speaker-independent with respect to the template set. "
        "These figures validate the implementation and show relative effects of design choices; they are not a claim about accuracy on natural human speech.", body)]

    nz = [["Test condition (SNR)", "Accuracy", "Mean real-time factor"]]
    for r in res["noise"]:
        label = "Clean" if r["snr_db"] == "clean" else f'{r["snr_db"]} dB'
        nz.append([label, f'{r["accuracy"]*100:.1f}%', f'{r["mean_rtf"]:.3f}'])
    story += [Spacer(1, 4), Paragraph("<b>Table 1 - Noise robustness</b> (white noise added to test clips only)", body), Spacer(1, 3),
              table(nz, [60 * mm, 40 * mm, W - 100 * mm])]

    ab = [["Design variant", "Clean", "10 dB SNR"]]
    for r in res["ablation"]:
        ab.append([r["variant"], f'{r["clean"]*100:.1f}%', f'{r["snr10"]*100:.1f}%'])
    story += [Spacer(1, 8), Paragraph("<b>Table 2 - Ablation study</b> (same templates, 180 test utterances per column)", body), Spacer(1, 3),
              table(ab, [W - 50 * mm, 25 * mm, 25 * mm])]

    rj = res["rejection"]
    conf_png = os.path.join(ROOT, "results", "confusion_clean.png")
    story += [Spacer(1, 8), Paragraph(
        "<b>Findings.</b> CMVN is the single most important normalisation step (10 dB accuracy "
        f"{res['ablation'][2]['snr10']*100:.0f}% without it versus {res['ablation'][0]['snr10']*100:.0f}% with it), and delta features give a modest gain ({res['ablation'][1]['snr10']*100:.1f}% without them). "
        "The band constraint changes accuracy only marginally while keeping cost bounded. Under noise the energy-based VAD is the weakest component: "
        f"disabling it raised 10 dB accuracy to {res['ablation'][3]['snr10']*100:.1f}% at the price of a higher real-time factor, which motivates "
        "a noise-robust VAD as future work. For out-of-vocabulary rejection, a threshold calibrated from the templates "
        f"({rj['threshold']:.2f}) accepted {rj['known_accepted']*100:.0f}% of {rj['n_known']} in-vocabulary utterances and rejected "
        f"{rj['unknown_rejected']*100:.0f}% of {rj['n_unknown']} utterances of a word that was not enrolled.", body)]

    story += [Spacer(1, 6), KeepTogether([Paragraph("<b>Figure 1 - Confusion matrix</b> (clean test set, unseen speakers)", body),
                                         Spacer(1, 2), Image(conf_png, width=88 * mm, height=88 * mm * 4.4 / 5.2)])]

    story += [Paragraph("Scope and Limitations", h1), Paragraph(
        "VoiceCmd-DTW is a small-vocabulary, template-matching recogniser: it recognises one isolated word per clip from the words that were enrolled, "
        "and its accuracy on real speech depends on enrolling several clear recordings per word from the same user. It is not a continuous-speech or "
        "open-vocabulary transcription system. Real recordings can be enrolled and evaluated with the app or the command-line tool without code changes.", body),
              Spacer(1, 8),
              Paragraph("<b>Note:</b> Reported results were produced by running scripts/evaluate_synthetic.py and the pytest suite on the submitted code; "
                        "they can be regenerated with the commands in README.md.", small)]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    doc = SimpleDocTemplate(args.out, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=13 * mm, bottomMargin=17 * mm,
                            title=f"{TITLE} - Academic Project Submission", author=f"{STUDENT} ({REG_NO})",
                            subject="Speech Recognition project submission")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print("wrote", args.out, f"({n_tests} tests passed)")


if __name__ == "__main__":
    main()
