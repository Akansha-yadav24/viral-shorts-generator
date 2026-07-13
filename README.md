# AI Viral Shorts Generator By Codingdidi

Automatically transform **long-form videos (20 minutes to 2+ hours)** into **high-quality YouTube Shorts, Instagram Reels, and TikTok clips** using **Python**, **Google Gemini**, and **Whisper**.

Instead of manually searching through hours of footage, this tool:

* Transcribes your video locally
* Uses Google's **FREE Gemini API** to identify the most engaging moments
* Cuts the clips automatically
* Generates subtitles
* Creates thumbnails
* Exports ready-to-upload vertical videos

---

# Demo Workflow

```text
             Long Video
                  │
                  ▼
        Extract Audio (FFmpeg)
                  │
                  ▼
   Transcribe using Faster-Whisper
                  │
                  ▼
     Analyze Transcript (Gemini API)
                  │
                  ▼
     Best Viral Moments Identified
                  │
                  ▼
    Validate & Merge Timestamps
                  │
                  ▼
      Export Vertical MP4 Clips
                  │
                  ▼
      Subtitles + Thumbnails
                  │
                  ▼
      clips.json + Output Folder
```

---

# Features

* AI-powered clip selection using Google Gemini
* Completely written in Python
* Uses FREE Gemini API
* Local speech transcription
* Batch processing
* Resume processing after interruption
* Automatic subtitle generation (.srt)
* Optional burned-in captions
* Automatic thumbnail generation
* 9:16 vertical Shorts export
* Original aspect ratio option
* Smart timestamp validation
* Duplicate removal
* Multi-threaded exporting
* Configurable through `config.json`
* Progress bars
* Structured logging
* Production-ready modular architecture

---

# Tech Stack

| Technology         | Purpose                          |
| ------------------ | -------------------------------- |
| Python 3.10+       | Main Programming Language        |
| Google Gemini API  | AI Clip Selection                |
| Faster Whisper     | Speech-to-Text                   |
| OpenAI Whisper     | Backup Transcription             |
| FFmpeg             | Audio Extraction & Video Cutting |
| ffmpeg-python      | Python FFmpeg Wrapper            |
| MoviePy            | Video Processing                 |
| python-dotenv      | Environment Variables            |
| tqdm               | Progress Bars                    |
| pathlib            | File Management                  |
| argparse           | CLI Interface                    |
| JSON               | Metadata Storage                 |
| ThreadPoolExecutor | Parallel Clip Export             |

---

# Folder Structure

```
viral-shorts-generator/

│
├── main.py
├── clip_generator.py
├── gemini_client.py
├── transcriber.py
├── video_editor.py
├── config.py
├── utils.py
│
├── config.json
├── requirements.txt
├── README.md
├── LICENSE
├── .env.example
│
```

---

# Requirements

* Python 3.10 or newer
* FFmpeg
* Google Gemini API Key
* 8GB RAM recommended
* CUDA GPU (Optional)

---

# Installation

## Clone Repository

```bash
git clone https://github.com/USERNAME/viral-shorts-generator.git

cd viral-shorts-generator
```

---

## Create Virtual Environment

Windows

```bash
python -m venv venv

venv\Scripts\activate
```

Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Install FFmpeg

### Windows

```powershell
choco install ffmpeg
```

or

Download

[https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)

Add FFmpeg to your PATH.

Verify

```bash
ffmpeg -version

ffprobe -version
```

---

### Ubuntu

```bash
sudo apt update

sudo apt install ffmpeg
```

---

### macOS

```bash
brew install ffmpeg
```

---

# Get Free Gemini API Key

Visit

[https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)

Generate a free API key.

Create

```
.env
```

```env
GEMINI_API_KEY=YOUR_API_KEY
```

Never commit `.env`.

---

# Usage

## Interactive Mode

```bash
python main.py
```

```
Enter Video Path:

Enter Output Folder:
```

---

## Single Video

```bash
python main.py \
--video "videos/tutorial.mp4" \
--output "output/"
```

---

## Batch Processing

```bash
python main.py \
--batch "videos/" \
--output "output/"
```

---

## Custom Config

```bash
python main.py \
--video sample.mp4 \
--output output \
--config config.json
```

---

## Force Reprocessing

```bash
python main.py \
--video sample.mp4 \
--output output \
--no-resume
```

---

# Configuration

Modify

```
config.json
```

Example

```json
{
    "gemini_model":"gemini-2.0-flash",
    "max_clip_seconds":90,
    "min_clip_seconds":20,
    "generate_subtitles":true,
    "burn_in_captions":false,
    "generate_thumbnails":true,
    "resume_supported":true,
    "output_width":1080,
    "output_height":1920
}
```

---

# Output

```
output/

└── tutorial/

    ├── clips/

    │     clip_001.mp4

    │     clip_001.srt

    │     clip_001_thumb.jpg

    │

    ├── clips.json

    │

    └── temp/
```

---

# Example clips.json

```json
[
    {
        "title":"Python Tips",
        "start":102.4,
        "end":181.3,
        "duration":78.9,
        "score":97,
        "reason":"Excellent hook",
        "filename":"clip_001.mp4"
    }
]
```

---

# Processing Pipeline

```
Input Video

↓

Extract Audio

↓

Speech Transcription

↓

Transcript Cleaning

↓

Gemini Analysis

↓

Timestamp Validation

↓

Merge Similar Clips

↓

Export MP4

↓

Generate Subtitle

↓

Generate Thumbnail

↓

Save Metadata
```

---

# Command Line Options

| Command       | Description               |
| ------------- | ------------------------- |
| `--video`     | Process a single video    |
| `--batch`     | Process an entire folder  |
| `--output`    | Output directory          |
| `--config`    | Custom configuration file |
| `--env`       | Custom environment file   |
| `--no-resume` | Disable transcript cache  |

---

# Supported Formats

### Input

* MP4
* MOV
* MKV
* AVI
* M4V
* WEBM

### Output

* MP4
* SRT
* JPG
* JSON

---

# Current Limitations

* Fixed center crop (no face tracking)
* No scene detection
* No speaker detection
* No semantic duplicate detection
* Subject to Gemini free-tier rate limits

---

# Roadmap

* Face Detection
* Auto Speaker Tracking
* AI Zoom Effects
* Emoji Captions
* B-Roll Suggestions
* Auto Hashtag Generator
* YouTube Shorts Upload
* Instagram Reels Upload
* TikTok Upload
* AI Title Generator
* AI Description Generator
* AI Thumbnail Generator
* Multi-language Subtitle Support
* Docker Support
* Web Dashboard
* Desktop GUI

---

# Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch

```bash
git checkout -b feature-name
```

3. Commit your changes

```bash
git commit -m "Added new feature"
```

4. Push to your branch

```bash
git push origin feature-name
```

5. Open a Pull Request

---

# License

This project is licensed under the **MIT License**.

---

# Disclaimer

This project uses the **Google Gemini API** for transcript analysis. Ensure that you have the necessary rights to process and republish any videos you use with this tool. The project is not affiliated with or endorsed by Google.
