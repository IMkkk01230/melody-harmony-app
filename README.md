
# 🎵 YouTube→メロディ＆ハモリ 自動生成（Streamlit Cloud用）

## 公開の超手順
1. このフォルダを **GitHubのPublicリポ** にアップ（`app.py`/`requirements.txt`/`packages.txt`）  
2. Streamlit Community Cloud にログイン → **New app**  
   - Repository：このリポジトリ  
   - Branch：main（または master）  
   - Main file：`app.py`  
   - Deploy  
3. 数分でURLが出ます。共有すればOK。

## 使い方
- 入力方法：「YouTubeのURL」または「ローカルファイル」
- キー/BPM/ハモリを選んで「生成」
- **MIDI / MusicXML / CSV** をDL

## 注意
- `packages.txt` に `ffmpeg` が必要（YouTube音声変換に必須）
- 体験版のため先頭 **60秒のみ解析**（`app.py`の`max_sec`を変更可）
- **権利のある音源だけ**解析してください。
