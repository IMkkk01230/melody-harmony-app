
import streamlit as st
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from io import BytesIO
from music21 import stream, note, key as m21key, meter, tempo, duration, midi, metadata
import os, tempfile, subprocess
from yt_dlp import YoutubeDL

st.set_page_config(page_title="Melody & Harmony from YouTube", page_icon="🎵")

# ------------------------
# Utilities
# ------------------------
def download_audio_from_youtube(url: str, max_seconds: int = 60) -> str:
    """
    YouTubeの音声を「先頭 max_seconds 秒だけ」FFmpegでストリーム抽出してWAV化します。
    → 全編ダウンロードしないので、無料枠でもタイムアウトしづらい方式。
    """
    import tempfile, os, subprocess
    from yt_dlp import YoutubeDL

    tmpdir = tempfile.mkdtemp(prefix="yt_")
    wav = os.path.join(tmpdir, "audio.wav")

    # 1) メタだけ取得（ダウンロードしない）
    ydl_opts = {
        "quiet": True,
        "noprogress": True,
        "skip_download": True,
        "format": "bestaudio/best",
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        # ベスト音声の実URLを選ぶ
        if "url" in info:  # 単一フォーマットの場合
            src_url = info["url"]
        else:
            # 複数フォーマットの場合は音声優先のbestを探す
            fmts = info.get("formats", [])
            audio_fmts = [f for f in fmts if f.get("acodec") != "none"]
            if not audio_fmts:
                raise RuntimeError("音声フォーマットが見つかりませんでした。")
            # 最大ビットレートのものを選ぶ
            src_url = sorted(audio_fmts, key=lambda f: f.get("abr") or 0, reverse=True)[0]["url"]

    # 2) FFmpegで先頭 max_seconds 秒だけ取り込み（44100Hz mono）
    #    ※HTTPストリームをそのまま読むので全編DL不要
    cmd = [
        "ffmpeg", "-y",
        "-ss", "0", "-t", str(int(max_seconds)),
        "-i", src_url,
        "-ac", "1", "-ar", "44100",
        wav
    ]
    ret = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)
    if ret.returncode != 0 or (not os.path.exists(wav)):
        # ログを出したい場合は以下を有効化:
        # print(ret.stderr.decode("utf-8", errors="ignore"))
        raise RuntimeError("YouTube音声の抽出に失敗しました。動画の公開設定/地域制限/年齢制限をご確認ください。")

    return wav

# ------------------------
# UI
# ------------------------
st.title("🎵 YouTubeからメロディ＆ハモリ自動生成（MVP）")
st.caption("※権利のある音源のみ解析してください。YouTube規約・著作権にご注意ください。")

with st.expander("使い方（超簡単）", expanded=False):
    st.write(
        "1) YouTubeのURLを貼る（自分の動画 or 許諾済み）\n"
        "2) キーとテンポを設定\n"
        "3) 生成ボタン → MIDI/MusicXML/CSVをDL"
    )

# すべてのウィジェットに「key」を付ける。動的切替は placeholder で一箇所に描画。
mode = st.radio(
    "入力方法",
    ["YouTubeのURL", "ローカルファイル（WAV/MP3など）"],
    key="mode_radio"
)

youtube_url = ""
uploaded = None
agree = False

input_placeholder = st.container()
with input_placeholder:
    if mode == "YouTubeのURL":
        youtube_url = st.text_input(
            "YouTube URL（自分の権利音源のみ）",
            value="",
            key="yt_url_input"
        )
        agree = st.checkbox(
            "このURLの音源は、私が権利を有する/許諾済みのものです。",
            value=False,
            key="rights_checkbox"
        )
    else:
        uploaded = st.file_uploader(
            "音声ファイルをアップロード",
            type=["wav","mp3","m4a","ogg","flac"],
            key="file_uploader"
        )

key_choice = st.selectbox(
    "キー（調）",
    ["C major","G major","D major","A major","E major","F major","Bb major","Eb major",
     "A minor","E minor","B minor","F# minor","D minor","G minor","C minor","F minor"],
    key="key_select"
)

bpm = st.number_input(
    "テンポ（BPM）",
    min_value=40, max_value=220, value=100, step=1,
    key="bpm_input"
)

harm_mode = st.selectbox(
    "ハモリの種類（ダイアトニック）",
    ["3度下（推奨）","3度上","6度上（やさしめ）"],
    key="harm_select"
)

harm_steps = {"3度下（推奨）": -2, "3度上": 2, "6度上（やさしめ）": 4}[harm_mode]

# 実行ボタンにもkeyを付与
if st.button("🎯 生成する", key="run_button"):
    try:
        if mode == "YouTubeのURL":
            if not youtube_url or not agree:
                st.error("URLを入力し、権利への同意にチェックしてください。")
                st.stop()
            wav_path = download_audio_from_youtube(youtube_url)
            y, sr = librosa.load(wav_path, sr=44100, mono=True)
        else:
            if uploaded is None:
                st.error("ファイルをアップロードしてください。")
                st.stop()
            y, sr = librosa.load(uploaded, sr=44100, mono=True)

        # 先頭60秒に制限（体験版）
        max_sec = 60.0
        if len(y)/sr > max_sec:
            y = y[:int(sr*max_sec)]
            st.info("体験版のため最初の60秒のみ解析しました。")

        hop = 512
        f0, vflag, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C6'),
            frame_length=2048,
            hop_length=hop
        )
        times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop)
        f0 = np.where(vflag, f0, np.nan)

        segs = segment_notes(times, f0, min_note_ms=120.0)
        if len(segs) == 0:
            st.warning("メロディを検出できませんでした。ボーカルがはっきりした音源でお試しください。")
            st.stop()

        score, mel, harm = notes_to_score(segs, bpm=bpm, key_text=key_choice, harmony_steps=harm_steps)

        df = pd.DataFrame([{
            "time_s": m["t"], "dur_s": m["dur"],
            "melody_note": m["note"], "melody_midi": m["midi"],
            "harmony_note": h["note"], "harmony_midi": h["midi"],
        } for m, h in zip(mel, harm)])

        st.success(f"検出ノート：{len(mel)} / キー：{key_choice} / ハモリ：{harm_mode}")
        st.dataframe(df.head(30))

        midi_bytes = export_midi(score)
        xml_bytes = export_musicxml(score)
        csv_bytes = df.to_csv(index=False).encode("utf-8-sig")

        st.download_button("⬇️ MIDI ダウンロード", data=midi_bytes, file_name="melody_harmony.mid", mime="audio/midi", key="dl_midi")
        st.download_button("⬇️ MusicXML ダウンロード", data=xml_bytes, file_name="melody_harmony.musicxml",
                           mime="application/vnd.recordare.musicxml+xml", key="dl_xml")
        st.download_button("⬇️ CSV ダウンロード", data=csv_bytes, file_name="melody_harmony.csv", mime="text/csv", key="dl_csv")

        st.caption("※第三者の著作物のダウンロードや再配布は不可です。")

    except Exception as e:
        st.error(f"エラー：{e}")
        st.stop()
