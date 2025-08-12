import os
import tempfile
import subprocess
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# === 解析系ライブラリ ===
import librosa
import librosa.display

# ページ設定
st.set_page_config(page_title="メロディ＆ハモリ解析", layout="wide")
st.title("🎵 メロディ＆ハモリ解析アプリ")

# --------- 共通ユーティリティ ---------
NOTE_NAMES_SHARP = np.array(
    ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
)

def hz_to_note_name(hz_array: np.ndarray) -> List[str]:
    """Hz -> 音名（C, C#, ...）"""
    midi = librosa.hz_to_midi(hz_array, round_=True)
    pcs = (midi % 12).astype(int)
    return NOTE_NAMES_SHARP[pcs].tolist()

def estimate_key_from_chroma(chroma: np.ndarray) -> str:
    """Krumhansl-Schmuckler風の簡易キー推定（メジャー/マイナー）"""
    # 12次元のテンプレート（正規化）
    major_profile = np.array(
        [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    )
    minor_profile = np.array(
        [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    )
    major_profile = major_profile / major_profile.sum()
    minor_profile = minor_profile / minor_profile.sum()

    avg = chroma.mean(axis=1)  # (12,)
    scores = []
    for shift in range(12):
        major_score = np.dot(np.roll(major_profile, shift), avg)
        minor_score = np.dot(np.roll(minor_profile, shift), avg)
        scores.append(("{} major".format(NOTE_NAMES_SHARP[shift]), major_score))
        scores.append(("{} minor".format(NOTE_NAMES_SHARP[shift]), minor_score))
    key, _ = max(scores, key=lambda x: x[1])
    return key

def chord_templates() -> Tuple[np.ndarray, List[str]]:
    """24コード（メジャー/マイナー）テンプレート"""
    T = []
    names = []
    triad_major = np.zeros(12); triad_major[[0, 4, 7]] = 1.0    # 1,3,5
    triad_minor = np.zeros(12); triad_minor[[0, 3, 7]] = 1.0    # 1,b3,5
    for i in range(12):
        T.append(np.roll(triad_major, i)); names.append(f"{NOTE_NAMES_SHARP[i]}")
        T.append(np.roll(triad_minor, i)); names.append(f"{NOTE_NAMES_SHARP[i]}m")
    T = np.stack(T, axis=0)  # (24,12)
    # 正規化
    T = T / (T.sum(axis=1, keepdims=True) + 1e-9)
    return T, names

def estimate_chords_from_chroma(chroma: np.ndarray, hop_time: float) -> pd.DataFrame:
    """フレーム毎に最尤コードを推定 → 連続区間をマージ"""
    T, names = chord_templates()  # (24,12)
    # 類似度：コサインに近い内積ベース
    # chroma shape: (12, frames)
    frames = chroma.shape[1]
    sims = T @ chroma  # (24, frames)
    idx = sims.argmax(axis=0)  # (frames,)
    # 連続区間マージ
    segments = []
    start = 0
    for i in range(1, frames):
        if idx[i] != idx[i-1]:
            segments.append((names[idx[start]], start, i-1))
            start = i
    segments.append((names[idx[start]], start, frames-1))
    # DataFrameへ
    rows = []
    for name, s, e in segments:
        t0 = s * hop_time
        t1 = (e+1) * hop_time
        rows.append({"chord": name, "start_sec": t0, "end_sec": t1, "duration": t1 - t0})
    df = pd.DataFrame(rows)
    return df

def analyze_audio(audio_path: str, limit_sec: float = 90.0):
    """メイン解析：主旋律ピッチ・音名、キー・コード推定"""
    y, sr = librosa.load(audio_path, sr=22050, mono=True, duration=limit_sec)

    # --- ピッチ推定（pYIN→失敗時はpiptrackにフォールバック） ---
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7")
        )
        # 無声音をNaN→除外
        pitch_hz = f0[~np.isnan(f0)]
        hop_time = librosa.frames_to_time(1, sr=sr, hop_length=512)  # pyin既定
        time_vec = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=512)
    except Exception:
        S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        idx = S.argmax(axis=0)
        pitch_hz = freqs[idx]
        hop_time = librosa.frames_to_time(1, sr=sr, hop_length=512)
        time_vec = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=512)

    # 音名（トップ10）
    note_names = hz_to_note_name(pitch_hz)
    top_notes = (
        pd.Series(note_names)
        .value_counts()
        .reset_index()
        .rename(columns={"index": "note", 0: "count"})
        .head(10)
    )

    # --- 和声（コード）推定 ---
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)  # (12, frames)
    key = estimate_key_from_chroma(chroma)
    chord_df = estimate_chords_from_chroma(chroma, hop_time=librosa.frames_to_time(1, sr=sr, hop_length=512))

    return {
        "sr": sr,
        "time_vec": time_vec,
        "pitch_hz": pitch_hz,
        "top_notes": top_notes,
        "key": key,
        "chords": chord_df,
    }

# --------- 入力UI ---------
mode = st.radio("入力方法を選択してください", ["ローカルファイル", "YouTube URL"], horizontal=True)
temp_dir = tempfile.mkdtemp()
audio_path = None

if mode == "ローカルファイル":
    up = st.file_uploader("音声ファイルをアップロード (mp3 / wav, 推奨は30〜90秒)", type=["mp3", "wav"])
    if up is not None:
        audio_path = os.path.join(temp_dir, up.name)
        with open(audio_path, "wb") as f:
            f.write(up.getbuffer())
        st.success("ファイルアップロード完了 ✅")

else:
    yt_url = st.text_input("YouTubeのURLを入力してください（公開・ログイン不要の動画を推奨）")
    consent = st.checkbox("私は著作権に注意して利用します")
    if st.button("ダウンロード開始"):
        if yt_url.startswith("http") and consent:
            try:
                # 先頭〜60秒のみ音声抽出（mp3 or wav）
                import yt_dlp
                ydl_opts = {
                    "format": "bestaudio/best",
                    "outtmpl": os.path.join(temp_dir, "yt_audio.%(ext)s"),
                    "download_ranges": {"download_ranges": ["00:00-00:59"]},
                    "postprocessors": [{
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": "mp3",
                        "preferredquality": "192",
                    }],
                    # 地域制限の軽減
                    "geo_bypass": True,
                    "nocheckcertificate": True,
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([yt_url])

                # mp3/wavを探す
                for f in os.listdir(temp_dir):
                    if f.endswith(".mp3") or f.endswith(".wav"):
                        audio_path = os.path.join(temp_dir, f)
                        break
                if audio_path:
                    st.success("YouTube音声取得完了 ✅")
                else:
                    st.error("音声変換に失敗しました。別のURLでお試しください。")
            except Exception as e:
                st.error(f"エラー: {e}")
        else:
            st.warning("正しいURLと同意チェックが必要です。")

# --------- 解析実行 ---------
if audio_path:
    st.audio(audio_path)
    if st.button("音階＆ハモリ解析開始"):
        with st.spinner("解析中…（最大90秒分を処理）"):
            result = analyze_audio(audio_path, limit_sec=90.0)

        col1, col2 = st.columns([1,1])

        # キーとトップノート
        with col1:
            st.subheader("推定キー（調）")
            st.success(result["key"])

            st.subheader("頻出ノート Top 10")
            st.dataframe(result["top_notes"], use_container_width=True)

        # コード概要
        with col2:
            st.subheader("推定コード上位（滞在時間）")
            top_chords = (
                result["chords"]
                .groupby("chord")["duration"]
                .sum()
                .sort_values(ascending=False)
                .reset_index()
                .head(10)
            )
            st.dataframe(top_chords, use_container_width=True)

        # ピッチのラインチャート
        st.subheader("主旋律ピッチ（Hz）")
        # NaNを除去して可視化
        pitch_df = pd.DataFrame({
            "time_sec": result["time_vec"][:len(result["pitch_hz"])],
            "pitch_hz": result["pitch_hz"]
        })
        pitch_df = pitch_df.replace([np.inf, -np.inf], np.nan).dropna()
        st.line_chart(pitch_df.set_index("time_sec"))

        # コードタイムライン
        st.subheader("コード進行タイムライン")
        # 10秒以上続くコードのみ強調（ノイズ抑制）
        tl = result["chords"].copy()
       tl["label"] = tl["chord"] + "  (" + tl["dur]()

