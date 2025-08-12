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

######################################################################
# 簡易コード推定（CQT + テンプレートマッチング）
######################################################################

CHORD_TEMPLATES = {
    # 12 半音（C=0）ベースのテンプレ
    "C":      [1,0,0,0,1,0,0,1,0,0,0,0],
    "Cm":     [1,0,0,1,0,0,0,1,0,0,0,0],
    "C#":     [0,1,0,0,0,1,0,0,1,0,0,0],
    "C#m":    [0,1,0,0,1,0,0,0,1,0,0,0],
    "D":      [0,0,1,0,0,0,1,0,0,1,0,0],
    "Dm":     [0,0,1,0,0,1,0,0,0,1,0,0],
    "Eb":     [0,0,0,1,0,0,0,1,0,0,1,0],
    "Ebm":    [0,0,0,1,0,0,1,0,0,0,1,0],
    "E":      [0,0,0,0,1,0,0,0,1,0,0,1],
    "Em":     [0,0,0,0,1,0,0,1,0,0,0,1],
    "F":      [1,0,0,0,0,1,0,0,0,1,0,0],  # F A C
    "Fm":     [1,0,0,0,0,1,0,0,1,0,0,0],  # F Ab C
    "F#":     [0,1,0,0,0,0,1,0,0,0,1,0],
    "F#m":    [0,1,0,0,0,0,1,0,0,1,0,0],
    "G":      [0,0,1,0,0,0,0,1,0,0,0,1],
    "Gm":     [0,0,1,0,0,0,0,1,0,0,1,0],
    "Ab":     [1,0,0,1,0,0,0,0,1,0,0,0],
    "Abm":    [1,0,0,1,0,0,0,0,0,1,0,0],
    "A":      [0,1,0,0,1,0,0,0,0,1,0,0],
    "Am":     [0,1,0,0,1,0,0,0,1,0,0,0],
    "Bb":     [0,0,1,0,0,1,0,0,0,0,1,0],
    "Bbm":    [0,0,1,0,0,1,0,0,1,0,0,0],
    "B":      [0,0,0,1,0,0,1,0,0,0,0,1],
    "Bm":     [0,0,0,1,0,0,1,0,0,0,1,0],
}

def est_chords(y: np.ndarray, sr: int, hop_length=4096) -> List[Dict]:
    """
    非常に簡易なコード推定：CQT → 12平均に畳み → テンプレート類似度最大のコード。
    戻り値: [{"start":sec, "end":sec, "chord":str, "dur":float}, ...]
    """
    C = np.abs(librosa.cqt(y, sr=sr, hop_length=hop_length, fmin=librosa.note_to_hz("C1"), n_bins=84, bins_per_octave=12))
    chroma = librosa.feature.chroma_cqt(C=C, sr=sr, hop_length=hop_length)
    # 時間方向メディアンで平滑
    chroma = medfilt(chroma, kernel_size=(1, 9))

    times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop_length)
    temps = {k: np.array(v, dtype=float) for k, v in CHORD_TEMPLATES.items()}
    for k in temps:
        temps[k] /= np.linalg.norm(temps[k]) + 1e-9

    indices = []
    for t in range(chroma.shape[1]):
        v = chroma[:, t]
        v = v / (np.linalg.norm(v) + 1e-9)
        sims = {k: float(v @ temps[k]) for k in temps}
        best = max(sims.items(), key=lambda x: x[1])[0]
        indices.append(best)

    # セグメント化
    segs = []
    if len(indices) == 0:
        return segs

    cur = indices[0]
    start = 0
    for i in range(1, len(indices)):
        if indices[i] != cur:
            segs.append({"start": float(times[start]), "end": float(times[i]), "chord": cur})
            cur = indices[i]
            start = i
    segs.append({"start": float(times[start]), "end": float(times[-1] if len(times) > 1 else 0.0), "chord": cur})

    # dur 付与
    for seg in segs:
        seg["dur"] = seg["end"] - seg["start"]
    return segs

######################################################################
# ハモリ生成（3rd / 6th の単純提案）
######################################################################

def suggest_harmony(midi_seq: np.ndarray, offset_semitone: int) -> np.ndarray:
    """ midi_seq: NaN 含む。NaN はそのまま NaN """
    harm = midi_seq.copy()
    mask = ~np.isnan(harm)
    harm[mask] = harm[mask] + offset_semitone
    return harm

def midi_to_note_name(m: float) -> str:
    if np.isnan(m):
        return ""
    n = int(round(m))
    return librosa.midi_to_note(n)

######################################################################
# 表示ラベル（ここが以前エラーだった所の安全版）
######################################################################

def _format_label(tl: Dict) -> str:
    chord = str(tl.get("chord", ""))
    # dur が数値 or 関数の両対応
    dur_value = None
    if "dur" in tl:
        try:
            dur_value = tl["dur"]() if callable(tl["dur"]) else tl["dur"]
        except Exception:
            dur_value = tl["dur"]

    if dur_value is None or dur_value == "":
        return chord
    try:
        dur_str = f"{float(dur_value):.2f}s"
    except Exception:
        dur_str = str(dur_value)
    return f"{chord} ({dur_str})"

######################################################################
# Streamlit UI
######################################################################

st.set_page_config(page_title="メロディ＆ハモリ解析アプリ", layout="wide")
st.title("🎵 メロディ＆ハモリ解析アプリ")

with st.sidebar:
    st.markdown("**使い方**")
    st.write("- ローカルファイル（mp3/wav）または YouTube URL を選択")
    st.write("- 解析開始を押すと、メロディ推定・コード推定・ハモリ提案を行います")
    st.caption("※ 著作権に配慮し、私的利用・研究目的のみにご利用ください")

# 入力切り替え
mode = st.radio("入力方法を選択してください", ["ローカルファイル", "YouTube URL"], index=0)
uploaded = None
wav_path = None

col_in1, col_in2 = st.columns([3, 2])
with col_in1:
    if mode == "ローカルファイル":
        uploaded = st.file_uploader("音声ファイルをアップロード (mp3 / wav / m4a など)", type=["mp3", "wav", "m4a", "aac", "flac"])
    else:
        yt_url = st.text_input("YouTubeのURLを入力してください", placeholder="https://www.youtube.com/watch?v=......")
with col_in2:
    agree = st.checkbox("私は著作権に注意して利用します", value=True, help="私的利用の範囲でご利用ください。")

btn = st.button("解析開始", type="primary", use_container_width=True)

if btn:
    if not agree:
        st.error("ご利用前に著作権への同意が必要です。")
        st.stop()

    # ---- 音声取得 ----
    st.info("🔊 音声を準備しています...")
    sr = 44100
    if mode == "ローカルファイル":
        if uploaded is None:
            st.error("ファイルを選択してください。")
            st.stop()
        wav_path = os.path.join(TMP_DIR, "input.wav")
        try:
            data = uploaded.read()
            save_bytes_to_wav(data, wav_path, sr=sr)
        except Exception as e:
            st.error(f"ファイルの読み込みに失敗: {e}")
            st.stop()
    else:
        if not yt_url.strip():
            st.error("YouTube のURLを入力してください。")
            st.stop()
        path, err = download_youtube_audio(yt_url.strip(), out_dir=TMP_DIR, sr=sr)
        if err:
            st.error(err)
            st.stop()
        wav_path = path

    # ---- 音声ロード ----
    try:
        y, sr = librosa.load(wav_path, sr=sr, mono=True)
        y = normalize_audio(y)
        duration = len(y) / sr
    except Exception as e:
        st.error(f"音声ロードに失敗: {e}")
        st.stop()

    st.success(f"音声準備OK（{duration:.2f} 秒）")

    # ---- メロディ推定 ----
    with st.spinner("🎼 メロディ（基本周波数）を推定中..."):
        mel = estimate_melody(y, sr)
    st.success("メロディ推定完了")

    # ---- コード推定 ----
    with st.spinner("🎹 コード（簡易）を推定中..."):
        chord_tl = est_chords(y, sr)
        for tl in chord_tl:
            tl["label"] = _format_label(tl)  # ★ここが安全版
    st.success(f"コード候補: {len(chord_tl)} セグメント")

    # ---- ハモリ提案 ----
    midi = f0_to_midi(mel.f0_hz)
    harm_up3 = suggest_harmony(midi, +3)   # 長短は気にせず 3半音
    harm_down3 = suggest_harmony(midi, -3)
    harm_up4 = suggest_harmony(midi, +4)
    harm_down5 = suggest_harmony(midi, -5)

    # ===== 画面出力 =====
    tabs = st.tabs(["概要", "メロディ可視化", "コードタイムライン", "ハモリ（提案）", "データ出力"])

    # --- 概要
    with tabs[0]:
        st.subheader("解析概要")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("全体長さ", f"{duration:.2f} s")
        with c2:
            voiced = np.mean(~np.isnan(mel.f0_hz)) * 100
            st.metric("有声率（ざっくり）", f"{voiced:.1f} %")
        with c3:
            st.metric("コードセグメント数", len(chord_tl))

        st.audio(wav_path)

    # --- メロディ可視化
    with tabs[1]:
        st.subheader("メロディ（pyin）の推定可視化")
        fig, ax = plt.subplots(figsize=(12, 4))
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=1024, hop_length=256)), ref=np.max)
        librosa.display.specshow(D, sr=sr, hop_length=256, x_axis="time", y_axis="hz", cmap="magma", ax=ax)
        ax.plot(mel.times, mel.f0_hz, color="cyan", lw=1.5, label="f0 (pyin)")
        ax.set_ylim(50, 2000)
        ax.legend(loc="upper right")
        ax.set_title("Spectrogram + f0")
        st.pyplot(fig, use_container_width=True)

        # 表示用テーブル
        df_mel = pd.DataFrame({
            "time": mel.times,
            "f0_hz": mel.f0_hz,
            "midi": f0_to_midi(mel.f0_hz),
            "note": [midi_to_note_name(v) for v in f0_to_midi(mel.f0_hz)]
        })
        st.dataframe(df_mel.head(500), use_container_width=True, height=300)

    # --- コードタイムライン
    with tabs[2]:
        st.subheader("コード（簡易）タイムライン")
        if len(chord_tl) == 0:
            st.info("コードセグメントが検出できませんでした。")
        else:
            df_ch = pd.DataFrame(chord_tl)
            df_ch["start(h:m:s)"] = df_ch["start"].apply(human_time)
            df_ch["end(h:m:s)"] = df_ch["end"].apply(human_time)
            st.dataframe(df_ch[["start", "end", "start(h:m:s)", "end(h:m:s)", "chord", "dur", "label"]],
                         use_container_width=True, height=350)

            # ざっくりの棒可視化
            fig, ax = plt.subplots(figsize=(12, 2 + 0.2 * len(chord_tl)))
            y0 = 0
            for tl in chord_tl:
                ax.barh(0, tl["dur"], left=tl["start"], height=0.4, color="#4CAF50")
                ax.text(tl["start"] + 0.02, 0, tl["label"], va="center", ha="left", color="white", fontsize=9)
            ax.set_yticks([])
            ax.set_xlabel("Time (s)")
            ax.set_xlim(0, duration + 0.1)
            ax.set_title("Chord Segments")
            st.pyplot(fig, use_container_width=True)

    # --- ハモリ提案
    with tabs[3]:
        st.subheader("ハモリ（提案）")
        st.caption("※ あくまで機械的な3度/4度/5度の単純提案です。実曲では不協和や避けたい進行が出るため、耳で最終調整してください。")

        choice = st.selectbox("インターバル", ["+3（上3度）", "-3（下3度）", "+4（上4度）", "-5（下5度）"], index=0)
        map_choice = {"+3（上3度）": harm_up3, "-3（下3度）": harm_down3, "+4（上4度）": harm_up4, "-5（下5度）": harm_down5}
        harm = map_choice[choice]

        df_harm = pd.DataFrame({
            "time": mel.times,
            "mel_note": [midi_to_note_name(v) for v in f0_to_midi(mel.f0_hz)],
            "harm_midi": harm,
            "harm_note": [midi_to_note_name(v) for v in harm]
        })
        st.dataframe(df_harm.head(500), use_container_width=True, height=350)

        # MIDIノート → 周波数 → シンセ的に試聴（簡易サイン波）
        def synth_from_midi(midis: np.ndarray, sr=22050) -> np.ndarray:
            dur = mel.times[-1] if len(mel.times) > 0 else 0.0
            n = int(dur * sr) + sr // 2
            out = np.zeros(n, dtype=np.float32)
            hop = int(round((mel.times[1] - mel.times[0]) * sr)) if len(mel.times) > 1 else 512
            pos = 0
            phase = 0.0
            for i in range(len(midis)):
                length = hop
                if i == len(midis) - 1 and len(mel.times) > 1:
                    length = int(((mel.times[i] + (mel.times[i] - mel.times[i-1])) - mel.times[i]) * sr)
                hz = librosa.midi_to_hz(midis[i]) if not np.isnan(midis[i]) else 0.0
                if hz <= 0:
                    pos += length
                    continue
                t = np.arange(length) / sr
                wave = 0.3 * np.sin(2 * np.pi * hz * t + phase)
                phase = (phase + 2 * np.pi * hz * length / sr) % (2 * np.pi)
                out[pos:pos+length] += wave.astype(np.float32)
                pos += length
            return normalize_audio(out)

      if st.button("（デモ）提案ハモリを合成して試聴"):
    with st.spinner("生成中..."):
        synth = synth_from_midi(harm, sr=22050)
        buf = io.BytesIO()
        sf.write(buf, synth, 22050, format="WAV")
        buf.seek(0)
    st.audio(buf, format="audio/wav")

    # --- データ出力
    with tabs[4]:
        st.subheader("データ出力")
        # メロディ
        df_mel_out = pd.DataFrame({
            "time": mel.times,
            "f0_hz": mel.f0_hz,
            "midi": f0_to_midi(mel.f0_hz),
            "note": [midi_to_note_name(v) for v in f0_to_midi(mel.f0_hz)]
        })
        csv_mel = df_mel_out.to_csv(index=False).encode("utf-8-sig")
        st.download_button("メロディCSVをダウンロード", csv_mel, file_name="melody.csv", mime="text/csv")

        # コード
        df_ch_out = pd.DataFrame(chord_tl)
        csv_ch = df_ch_out.to_csv(index=False).encode("utf-8-sig")
        st.download_button("コードタイムラインCSVをダウンロード", csv_ch, file_name="chords.csv", mime="text/csv")

        json_all = {
            "duration_sec": float(duration),
            "melody": df_mel_out.to_dict(orient="records"),
            "chords": chord_tl,
        }
        st.download_button("JSONをダウンロード", json.dumps(json_all, ensure_ascii=False, indent=2).encode("utf-8"),
                           file_name="analysis.json", mime="application/json")

else:
    st.caption("「解析開始」を押すと処理が始まります。YouTube で失敗する場合はローカルファイルでお試しください。")
