"""VoiceCmd-DTW: isolated-word speech recognition with MFCC features and
Dynamic Time Warping (DTW), implemented from scratch with NumPy/SciPy."""

from .audio_io import AudioError, NoSpeechError, load_wav, preprocess, resample, validate_audio
from .dtw import DTWResult, dtw
from .evaluation import EvalReport, evaluate
from .features import FeatureConfig, extract_features
from .recognizer import DTWRecognizer, Recognition
from .vad import detect_speech

__version__ = "1.0.0"

__all__ = [
    "AudioError", "NoSpeechError", "load_wav", "preprocess", "resample",
    "validate_audio", "DTWResult", "dtw", "EvalReport", "evaluate",
    "FeatureConfig", "extract_features", "DTWRecognizer", "Recognition",
    "detect_speech", "__version__",
]
