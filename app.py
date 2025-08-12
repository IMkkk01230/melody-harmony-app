
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
def download_audio_from_youtube(url: str) -> str:
    """
    Download audio from YouTube and convert to 44.1kHz mono WAV via ffmpeg.
    Use only for your own/authorized content.
    """
    tmpdir = tempfile.mkdtemp(prefix="yt_")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(tmpdir, "audio.%(ext)s"),
        "quiet": True,
        "noprogress": True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    files = [os.path.join(tmpdir, f) for f in os.listdir(tmpdir)]
    if not files:
        raise RuntimeError("音声の取得に失敗しました。URL/公開設定をご確認ください。")
    src = files[0]
    wav = os.path.join(tmpdir, "audio.wav")
    cmd = ["ffmpeg","-y","-i",src,"-ac","1","-ar","44100",wav]
    ret = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if ret.returncode != 0:
        raise RuntimeError("ffmpegの変換に失敗しました（ログを確認）。")
    return wav

def segment_notes(times, f0, min_note_ms=120.0):
    min_note_s = min_note_ms / 1000.0
    segs = []
    cur_pitch = None
    cur_start = None

    def flush(idx):
        nonlocal cur_pitch, cur_start
        if cur_pitch is None or cur_start is None:
            return
        start_t = cur_start
        end_t = times[idx] if idx < len(times) else times[-1]
        dur = max(0.0, end_t - start_t)
        if dur >= min_note_s:
            segs.append({"t": float(start_t), "dur": float(dur), "hz": float(cur_pitch)})
        cur_pitch = None
        cur_start = None

    for i, hz in enumerate(f0):
        if np.isnan(hz):
            if cur_pitch is not None:
                flush(i)
            continue
        midi_pitch = np.round(librosa.hz_to_midi(hz))
        if cur_pitch is None:
            cur_pitch = librosa.midi_to_hz(midi_pitch)
            cur_start = times[i]
        else:
            prev_midi = librosa.hz_to_midi(cur_pitch)
            if abs(midi_pitch - prev_midi) <= 0.5:
                pass
            else:
                flush(i)
                cur_pitch = librosa.midi_to_hz(midi_pitch)
                cur_start = times[i]
    flush(len(times) - 1)
    return segs

def build_key(key_text):
    name, mode = key_text.split()
    return m21key.Key(name, mode)

def diatonic_shift_midi(midi_pitch, m21_key, steps):
    n = note.Note()
    n.pitch.midi = midi_pitch
    deg = m21_key.getScaleDegreeFromPitch(n.pitch)
    if deg is None:
        candidates = []
        for d in range(1, 8):
            p = m21_key.pitchFromDegree(d)
            for off in [-24,-12,0,12,24]:
                nn = note.Note(p)
                nn.pitch.midi += off
                candidates.append(nn.pitch.midi)
        n.pitch.midi = min(candidates, key=lambda x: abs(x - midi_pitch))
        deg = m21_key.getScaleDegreeFromPitch(n.pitch)
    target_deg = ((deg - 1 + steps) % 7) + 1
    tp = m21_key.pitchFromDegree(target_deg)
    tn = note.Note(tp)
    while tn.pitch.midi > midi_pitch + 7:
        tn.pitch.midi -= 12
    while tn.pitch.midi < midi_pitch - 12:
        tn.pitch.midi += 12
    return tn.pitch.midi

def notes_to_score(notes, bpm, key_text, harmony_steps):
    sc = stream.Score()
    sc.insert(0, metadata.Metadata())
    sc.metadata.title = "Melody & Harmony (YouTube)"
    kp = build_key(key_text)
    ts = meter.TimeSignature('4/4')
    tp = tempo.MetronomeMark(number=bpm)

    part_m = stream.Part(id="Melody")
    part_h = stream.Part(id="Harmony")
    for p in (part_m, part_h):
        p.insert(0, kp); p.insert(0, ts); p.insert(0, tp)

    sec_per_beat = 60.0 / max(bpm, 1.0)
    mel_rows, harm_rows = [], []

    for seg in notes:
        midi_pitch = int(np.round(librosa.hz_to_midi(seg["hz"])))
        dur_q = max(seg["dur"] / sec_per_beat, 0.25)
        nm = note.Note()
        nm.pitch.midi = midi_pitch
        nm.duration = duration.Duration(dur_q)
        part_m.append(nm)
        mel_rows.append({"t": seg["t"], "dur": seg["dur"], "note": nm.nameWithOctave, "midi": midi_pitch})

        hm = note.Note()
        hm.pitch.midi = diatonic_shift_midi(midi_pitch, kp, harmony_steps)
        hm.duration = duration.Duration(dur_q)
        part_h.append(hm)
        harm_rows.append({"t": seg["t"], "dur": seg["dur"], "note": hm.nameWithOctave, "midi": hm.pitch.midi})

    sc.append(part_m); sc.append(part_h)
    return sc, mel_rows, harm_rows

def export_midi(score):
    mf = midi.translate.streamToMidiFile(score)
    bio = BytesIO()
    mf.open(bio); mf.write(); mf.close()
    bio.seek(0); return bio.read()

def export_musicxml(score):
    fp = score.write('musicxml')
    with open(fp,'rb') as f: return f.read()

# ------------------------
# UI
# ------------------------
st.title("🎵 YouTubeからメロディ＆ハモリ自動生成（MVP）")
st.caption("※権利のある音源のみ解析してください。YouTube規約・著作権にご注意ください。")

mode = st.radio("入力方法", ["YouTubeのURL", "ローカルファイル（WAV/MP3など）"])

youtube_url = ""
uploaded = None
if mode == "YouTubeのURL":
    youtube_url = st.text_input("YouTube URL（自分の権利音源のみ）", "")
    agree = st.checkbox("このURLの音源は、私が権利を有する/許諾済みのものです。", value=False)
else:
    uploaded = st.file_uploader("音声ファイルをアップロード", type=["wav","mp3","m4a","ogg","flac"])

key_choice = st.selectbox("キー（調）", 
    ["C major","G major","D major","A major","E major","F major","Bb major","Eb major",
     "A minor","E minor","B minor","F# minor","D minor","G minor","C minor","F minor"])
bpm = st.number_input("テンポ（BPM）", min_value=40, max_value=220, value=100, step=1)
harm_mode = st.selectbox("ハモリ（ダイアトニック）", ["3度下（推奨）","3度上","6度上（やさしめ）"])
harm_steps = {"3度下（推奨）": -2, "3度上": 2, "6度上（やさしめ）": 4}[harm_mode]

if st.button("🎯 生成する"):
    try:
        if mode == "YouTubeのURL":
            if not youtube_url or not agree:
                st.error("URLを入力し、権利への同意にチェックしてください。"); st.stop()
            wav_path = download_audio_from_youtube(youtube_url)
            y, sr = librosa.load(wav_path, sr=44100, mono=True)
        else:
            if uploaded is None:
                st.error("ファイルをアップロードしてください。"); st.stop()
            y, sr = librosa.load(uploaded, sr=44100, mono=True)

        max_sec = 60.0
        if len(y)/sr > max_sec:
            y = y[:int(sr*max_sec)]
            st.info("体験版のため最初の60秒のみ解析しました。")

        hop = 512
        f0, vflag, _ = librosa.pyin(y,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C6'),
            frame_length=2048, hop_length=hop)
        times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop)
        f0 = np.where(vflag, f0, np.nan)

        segs = segment_notes(times, f0, min_note_ms=120.0)
        if len(segs) == 0:
            st.warning("メロディを検出できませんでした。ボーカルがはっきりした音源でお試しください。"); st.stop()

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

        st.download_button("⬇️ MIDI ダウンロード", data=midi_bytes, file_name="melody_harmony.mid", mime="audio/midi")
        st.download_button("⬇️ MusicXML ダウンロード", data=xml_bytes, file_name="melody_harmony.musicxml",
                           mime="application/vnd.recordare.musicxml+xml")
        st.download_button("⬇️ CSV ダウンロード", data=csv_bytes, file_name="melody_harmony.csv", mime="text/csv")

        st.caption("※第三者の著作物のダウンロードや再配布は不可です。")

    except Exception as e:
        st.error(f"エラー：{e}")
        st.stop()
