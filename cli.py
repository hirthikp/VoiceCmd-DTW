#!/usr/bin/env python3
"""Command-line interface for VoiceCmd-DTW.

Examples
--------
  python cli.py generate-demo data/demo
  python cli.py enroll data/demo/enroll --out model.npz
  python cli.py recognize model.npz data/demo/test/go/spk20_rep0.wav
  python cli.py evaluate model.npz data/demo/test
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from voicecmd import AudioError, DTWRecognizer, FeatureConfig, evaluate, load_wav
from voicecmd.datasets import enroll_dir, load_labelled_dir
from voicecmd.synthetic import write_demo_dataset


def cmd_generate(args):
    counts = write_demo_dataset(args.folder)
    print("Wrote demo clips:", counts)


def cmd_enroll(args):
    rec = DTWRecognizer(FeatureConfig(), band=args.band, k=args.k)
    n = enroll_dir(rec, args.folder)
    rec.save(args.out)
    print(f"Enrolled {n} clips ({len(rec.words)} words: {', '.join(rec.words)}) -> {args.out}")


def cmd_recognize(args):
    rec = DTWRecognizer.load(args.model)
    if args.threshold is not None:
        rec.reject_threshold = args.threshold
    y, sr = load_wav(args.wav)
    res = rec.recognize(y, sr)
    if args.json:
        print(json.dumps(res.to_dict(), indent=2))
        return
    status = "accepted" if res.accepted else "REJECTED (distance above threshold)"
    print(f"Word      : {res.word}  [{status}]")
    print(f"Distance  : {res.distance:.3f}   margin {res.margin:.2f}")
    print(f"Audio     : {res.audio_duration_s:.2f}s  processed in {res.elapsed_s*1000:.1f} ms  (RTF {res.rtf:.3f})")
    print("Ranking   : " + ", ".join(f"{w} {d:.2f}" for w, d in res.ranking))


def cmd_evaluate(args):
    rec = DTWRecognizer.load(args.model)
    rep = evaluate(rec, load_labelled_dir(args.folder))
    print(f"Accuracy: {rep.accuracy*100:.1f}%  ({rep.n_correct}/{rep.n}, failed {rep.n_failed})  mean RTF {rep.mean_rtf:.3f}")
    for w, m in rep.per_class().items():
        print(f"  {w:<8} precision {m['precision']:.2f}  recall {m['recall']:.2f}  n={m['support']}")


def main(argv=None):
    p = argparse.ArgumentParser(description="VoiceCmd-DTW isolated-word recogniser")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate-demo", help="create synthetic demo WAV clips")
    g.add_argument("folder")
    g.set_defaults(fn=cmd_generate)

    e = sub.add_parser("enroll", help="build a model from root/<word>/*.wav")
    e.add_argument("folder")
    e.add_argument("--out", default="model.npz")
    e.add_argument("--band", type=float, default=0.3)
    e.add_argument("--k", type=int, default=1)
    e.set_defaults(fn=cmd_enroll)

    r = sub.add_parser("recognize", help="recognise one WAV file")
    r.add_argument("model")
    r.add_argument("wav")
    r.add_argument("--threshold", type=float, default=None)
    r.add_argument("--json", action="store_true")
    r.set_defaults(fn=cmd_recognize)

    v = sub.add_parser("evaluate", help="score a model on root/<word>/*.wav")
    v.add_argument("model")
    v.add_argument("folder")
    v.set_defaults(fn=cmd_evaluate)

    args = p.parse_args(argv)
    try:
        args.fn(args)
    except (AudioError, FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
