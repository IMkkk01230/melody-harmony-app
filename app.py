# app.py
# -*- coding: utf-8 -*-
# メロディ＆ハモリ解析アプリ（YouTube / ローカル対応）
# 依存: streamlit, numpy, pandas, librosa, soundfile, matplotlib, yt_dlp, scipy

import os
import io
import sys
import json
import math
import time
import shutil
import warnings
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import soundfile as sf
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.signal import medfilt

# YouTube ダウンロード
try:
    from yt_dlp import YoutubeDL
    _YTDLP_AVAILABLE = True
except Exception:
    _YTDLP_AVAILABLE = False

warnings.filterwarnings("ignore")

######################################################################
# ユーティリティ
######################################################################

TMP_DIR = "tmp_inputs"
os.makedirs(TMP_DIR, exist_ok=True)

def human_time(sec: float) -> str:
    if sec is None or np.isnan(sec):
        return "-"
    m = int(sec // 60)
    s = sec - 60 * m
    return f"{m:02d}:{s:05.2f}"

def save_bytes_to_wav(b: bytes, path: str, sr=44100):
    y, sr0 = sf.read(io.BytesIO(b), dtype="float32", always_2d=False)
    if sr0 != sr:
        y = librosa.resample(y, orig_sr=sr0, target_sr=sr)
    sf.write(path, y, sr)

def normalize_audio(y, peak=0.98):
    m = np.max(np.abs(y)) + 1e-9
    return y * (peak / m)

######################################################################
# YouTube から音声抽出
######################################################################

def download_youtube_audio(url: str, out_dir: str = TMP_DIR, sr=44100) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns: (wav_path, err_message)
    """
    if not _YTDLP_AVAILABLE:
        return None, "yt_dlp が見つかりません。ローカルファイルでお試しください。"

    os.makedirs(out_dir, exist_ok=True)
    wav_path = os.path.join(out_dir, "yt_audio.wav")

    # 一旦は m4a/opus 等で落とし、ffmpeg で wav 化（Streamlit Cloud でも動く）
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(out_dir, "audio.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "nocheckcertificate": True,
        "retries": 2,
        "geo_bypass": True,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "0"
        }],
        "postprocessor_args": [
            "-ar", str(sr),
            "-ac", "1"
        ],
    }

    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        # 出力は audio.wav のはず
        cand = os.path.join(out_dir, "audio.wav")
        if not os.path.exists(cand):
            return None, "ダウンロードに失敗しました（bot対策・年齢制限・地域制限などの可能性）。ローカルファイルでお試しください。"
        # 統一名に
        shutil.move(cand, wav_path)
        return wav_path, None
    except Exception as e:
        return None, f"YouTube ダウンロードでエラー: {e}"

######################################################################
# メロディ推定（pyin）
######################################################################

@dataclass
class MelodyResult:
    times: np.ndarray
    f0_hz: np.ndarray           # NaN を含む
    confidence: np.ndarray      # 0..1

def estimate_melody(y: np.ndarray, sr: int) -> MelodyResult:
    # 低域ノイズ対策で軽くHPF
    y = librosa.effects.preemphasis(y)
    fmin = librosa.note_to_hz("C2")
    fmax = librosa.note_to_hz("C7")

    f0, vflag, _ = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr, frame_length=2048, hop_length=256)
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=256)
    conf = vflag.astype(float)
    return MelodyResult(times=times, f0_hz=f0, confidence=conf)

def f0_to_midi(f0_hz: np.ndarray) -> np.ndarray:
    midi = librosa.hz_to_midi(f0_hz)
    return midi

################################################################
