import streamlit as st
import os
import tempfile
import subprocess
from yt_dlp import YoutubeDL

# ページ設定
st.set_page_config(page_title="メロディ＆ハモリ解析", layout="centered")
st.title("🎵 メロディ＆ハモリ解析アプリ")

# 入力方法選択
mode = st.radio("入力方法を選択してください", ["ローカルファイル", "YouTube URL"])

# 一時フォルダ
temp_dir = tempfile.mkdtemp()

# 音声ファイル変数
audio_path = None

if mode == "ローカルファイル":
    uploaded_file = st.file_uploader("音声ファイルをアップロード (mp3 / wav)", type=["mp3", "wav"])
    if uploaded_file is not None:
        audio_path = os.path.join(temp_dir, uploaded_file.name)
        with open(audio_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("ファイルアップロード完了 ✅")

elif mode == "YouTube URL":
    yt_url = st.text_input("YouTubeのURLを入力してください")
    consent = st.checkbox("私は著作権に注意して利用します")
    if st.button("ダウンロード開始"):
        if yt_url.startswith("http") and consent:
            try:
                ydl_opts = {
                    "format": "bestaudio/best",
                    "outtmpl": os.path.join(temp_dir, "yt_audio.%(ext)s"),
                    "download_ranges": {"download_ranges": ["00:00-00:59"]},
                    "postprocessors": [{
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": "wav",
                        "preferredquality": "192",
                    }],
                }
                with YoutubeDL(ydl_opts) as ydl:
                    ydl.download([yt_url])

                # wavファイルを探す
                for f in os.listdir(temp_dir):
                    if f.endswith(".wav"):
                        audio_path = os.path.join(temp_dir, f)
                        break
                if audio_path:
                    st.success("YouTube音声取得完了 ✅")
                else:
                    st.error("音声変換に失敗しました。")
            except Exception as e:
                st.error(f"エラー: {e}")
        else:
            st.warning("正しいURLと同意チェックが必要です。")

# 音声解析処理（ここは仮のダミー）
if audio_path and st.button("音階＆ハモリ解析開始"):
    # 実際は音階解析処理をここに追加
    st.info(f"解析対象: {audio_path}")
    st.success("解析完了（デモ）")
