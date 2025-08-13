import io
import os
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st

# librosa まわり
import librosa
import librosa.display
import soundfile as sf  # BytesIO に WAV を書くため


# =============== ページ設定 ==================
st.set_page_config(
    page_title="メロディ & ハモリ解析アプリ",
    page_icon="🎵",
    layout="centered",
)


# =============== ユーティリティ ===============
@st.cache_data(show_spinner=False)
def _load_audio_from_bytes(data: bytes, sr: int = 22050) -> Tuple[np.ndarray, int]:
    y, sr = librosa.load(io.BytesIO(data), sr=sr, mono=True)
    return y, sr


@st.cache_data(show_spinner=False)
def _load_audio_from_path(path: str, sr: int = 22050) -> Tuple[np.ndarray, int]:
    y, sr = librosa.load(path, sr=sr, mono=True)
    return y, sr


def _estimate_key_from_chroma(chroma: np.ndarray) -> str:
    """平均クロマから大雑把に調性を推定（簡易版）"""
    # メジャー / マイナーのテンプレート相関で判定
    # 12音クロマのメジャー/マイナーコードの平均形に近い方を採用
    # （厳密ではないが、デモには十分）
    major_template = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1], float)
    minor_template = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0], float)

    mean_chroma = chroma.mean(axis=1)
    best = None
    best_score = -1e9
    names = np.array(["C", "C#", "D", "D#", "E", "F",
                      "F#", "G", "G#", "A", "A#", "B"])

    for shift in range(12):
        rot_major = np.roll(major_template, shift)
        rot_minor = np.roll(minor_template, shift)
        s_major = float(np.dot(mean_chroma, rot_major))
        s_minor = float(np.dot(mean_chroma, rot_minor))

        if s_major >= s_minor and s_major > best_score:
            best = f"{names[shift]} major"
            best_score = s_major
        if s_minor > s_major and s_minor > best_score:
            best = f"{names[shift]} minor"
            best_score = s_minor
    return best or "Unknown"


def _chord_name_from_chroma_vector(v: np.ndarray) -> str:
    """1フレームのクロマベクトルからメジャー/マイナーコード名を簡易推定"""
    names = np.array(["C", "C#", "D", "D#", "E", "F",
                      "F#", "G", "G#", "A", "A#", "B"])

    major_template = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1], float)
    minor_template = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0], float)

    best = None
    best_score = -1e9
    for shift in range(12):
        s_major = float(np.dot(v, np.roll(major_template, shift)))
        s_minor = float(np.dot(v, np.roll(minor_template, shift)))
        if s_major >= s_minor and s_major > best_score:
            best = f"{names[shift]}"
            suffix = ""
            best_score = s_major
        if s_minor > s_major and s_minor > best_score:
            best = f"{names[shift]}"
            suffix = "m"
            best_score = s_minor
    return (best or "N") + (suffix if best else "")


def _build_chord_timeline(
    y: np.ndarray, sr: int, hop_length: int = 512
) -> Tuple[pd.DataFrame, float, float, str]:
    """
    ビートごとに簡易コード推定してタイムラインに整形。
    戻り値: (DataFrame, 推定テンポ, 総時間, 推定キー)
    """
    duration = len(y) / sr

    # テンポ/ビート
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
    beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=hop_length)
    if len(beat_times) == 0:
        # ビートが取れない場合は一定間隔で代替
        beat_times = np.linspace(0, duration, num=1 + int(duration // 0.5))

    # クロマ（CQT）
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    key_name = _estimate_key_from_chroma(chroma)

    # 各ビート区間の平均クロマからコード推定
    frames = np.arange(chroma.shape[1])
    times = librosa.frames_to_time(frames, sr=sr)

    segments: List[Dict] = []
    for i in range(len(beat_times)):
        start = float(beat_times[i])
        end = float(beat_times[i + 1] if i + 1 < len(beat_times) else duration)
        mask = (times >= start) & (times < end)
        if not np.any(mask):
            # フレームなし区間はスキップ
            continue
        mean_v = chroma[:, mask].mean(axis=1)
        name = _chord_name_from_chroma_vector(mean_v)
        dur = max(0.0, end - start)
        label = f"{name} ({dur:.2f}s)"  # ← 文字列連結はこの形が安全
        segments.append(
            {
                "start": start,
                "end": end,
                "dur": dur,
                "chord": name,
                "label": label,
            }
        )

    df = pd.DataFrame(segments)
    if not df.empty:
        df = df[["start", "end", "dur", "chord", "label"]]
    return df, float(tempo), float(duration), key_name


def _mix_harmony(y: np.ndarray, sr: int, shift_semitones: float = 3.0, gain: float = 0.6) -> np.ndarray:
    """
    原音に対してピッチシフトしたトラックを重ねる簡易ハモリ（長3度上をデフォルト）。
    """
    harm = librosa.effects.pitch_shift(y, sr=sr, n_steps=shift_semitones)
    # レベル調整してミックス
    mix = y + gain * harm
    # クリップ防止
    mx = np.max(np.abs(mix)) + 1e-9
    if mx > 1.0:
        mix = mix / mx
    return mix


def _audio_to_wav_bytes(y: np.ndarray, sr: int) -> bytes:
    """
    メモリ上（BytesIO）に WAV で書き出して bytes を返す。
    """
    bio = io.BytesIO()
    sf.write(bio, y, sr, format="WAV")
    bio.seek(0)
    return bio.getvalue()


# =============== メインUI ===============
st.title("🎵 メロディ & ハモリ解析アプリ")

st.caption(
    "MP3/WAV を解析して、**テンポ・推定キー・簡易コード進行**を表示します。"
    " デモとして、原音に 3 度上のハモリを重ねた音も試聴できます。"
)

with st.expander("利用上の注意（著作権）", expanded=False):
    st.markdown(
        "- 著作権に配慮し、**自分で利用可能な音源**のみアップロード/解析してください。\n"
        "- YouTube からのダウンロードは動画側の制限や地域・年齢制限、reCAPTCHA などの理由で失敗することがあります。\n"
        "- まずは **ローカルファイル** の解析が確実に動作します。"
    )

# 入力方法
input_mode = st.radio(
    "入力方法を選択してください",
    options=["ローカルファイル", "YouTube URL"],
    horizontal=True,
)

audio_bytes: bytes | None = None
audio_name: str = ""

if input_mode == "ローカルファイル":
    up = st.file_uploader("音声ファイルをアップロード（MP3 / WAV）", type=["mp3", "wav"])
    if up is not None:
        audio_bytes = up.getvalue()
        audio_name = up.name

else:
    url = st.text_input("YouTube の URL を入力してください")
    agree = st.checkbox("私は著作権に注意して利用します", value=False)
    if url and agree and st.button("YouTube から音声を取得（任意機能）"):
        with st.spinner("取得中...（失敗する場合はローカルファイルでお試しください）"):
            # 依存と環境の都合で、ここでは実処理を省略 or プレースホルダ。
            # 実装したい場合は yt_dlp + ffmpeg を利用してください。
            st.warning("このデモでは YouTube ダウンロードを無効化しています。ローカルファイルをご利用ください。")

# 解析ボタン
if audio_bytes is not None and st.button("音階 & ハモリ解析開始"):
    try:
        with st.spinner("読み込み中..."):
            y, sr = _load_audio_from_bytes(audio_bytes, sr=22050)

        with st.spinner("特徴量解析中..."):
            df, tempo, duration, key_name = _build_chord_timeline(y, sr)

        # ---- 解析結果の表示 ----
        st.subheader("基本情報")
        c1, c2, c3 = st.columns(3)
        c1.metric("長さ", f"{duration:.2f} s")
        c2.metric("推定テンポ", f"{tempo:.1f} BPM")
        c3.metric("推定キー", key_name)

        st.subheader("簡易コード進行（ビート単位）")
        if df.empty:
            st.info("コードを推定できませんでした。無音区間が長い・極端に短い音源などの可能性があります。")
        else:
            st.dataframe(
                df.style.format({"start": "{:.2f}", "end": "{:.2f}", "dur": "{:.2f}"}),
                use_container_width=True,
                height=min(480, 40 + 28 * len(df)),
            )

            # CSV ダウンロード
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "解析結果（CSV）をダウンロード",
                data=csv,
                file_name=(Path(audio_name).stem if audio_name else "analysis") + "_chords.csv",
                mime="text/csv",
            )

        # ---- 試聴：元音源 / 簡易ハモリ ----
        st.subheader("試聴")
        col_a, col_b = st.columns(2)

        with col_a:
            st.caption("原音（モノラル）")
            st.audio(_audio_to_wav_bytes(y, sr), format="audio/wav")

        with col_b:
            st.caption("（デモ）提案ハモリを合成して試聴")
            mix = _mix_harmony(y, sr, shift_semitones=3.0, gain=0.6)
            st.audio(_audio_to_wav_bytes(mix, sr), format="audio/wav")

    except Exception as e:
        st.error(f"解析に失敗しました: {type(e).__name__}: {e}")

elif audio_bytes is None and input_mode == "ローカルファイル":
    st.info("MP3 / WAV を選んで「音階 & ハモリ解析開始」を押してください。")
