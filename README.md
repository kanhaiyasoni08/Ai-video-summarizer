# AI Video Summarizer

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)](https://streamlit.io/)
[![Whisper](https://img.shields.io/badge/Transcription-Whisper-orange)](https://openai.com/research/whisper)

An intelligent AI-powered video summarizer that **transcribes video audio**, **extracts key points**, and generates **concise summaries**, **highlights**, and optional **visual annotations**, all through an intuitive **Streamlit UI**.

---

## 🌟 Live Demo Preview

*(Add link to a deployed version here if available)*

---

## 🎯 The Problem

Video content is everywhere — but watching full videos takes time. Whether it’s lectures, webinars, interviews, or tutorials, users struggle with:

* ⏱️ **Time Consumption:** Watching long videos just to extract key insights.
* 🧠 **Cognitive Load:** Hard to remember or note the important parts.
* 📝 **Manual Effort:** Manual transcription and summarization are slow and tedious.

This project solves these challenges by using AI to automatically transcribe videos, extract key moments, and produce readable summaries, saving users time and effort.

---

## ✨ Key Features

* 📥 **Video Upload:** Upload video files (`.mp4`, `.avi`, etc.) directly in the UI.
* 🗣️ **Accurate Transcription:** Uses **OpenAI’s Whisper** model to convert the audio track into text.
* 🧠 **AI-Generated Summary:** Uses **Gemini** (or your chosen LLM) to produce a concise summary of the video’s content.
* 📌 **Highlights & Key Moments:** Breaks transcripts into chunks and identifies the most important segments.
* 🔍 **Optional Visual Analysis:**

  * **Frame Captioning** using `BLIP`
  * **OCR** via `EasyOCR` to extract text appearing in video frames.
  * Object/scene insights using OpenCV.
* 🖼️ **Streamlit UI:** Interactive frontend where users upload files and view summaries.

---

## 🏗️ System Architecture

The system is designed to process videos in a stepwise workflow combining **transcription**, **chunking**, **captioning**, and **AI summarization**.

1. **Video Input**
   User uploads a video file via the Streamlit UI.

2. **Audio Extraction & Whisper Transcription**
   The app extracts the audio track and uses Whisper to generate a high-quality transcript.

3. **Chunking & Preprocessing**
   The transcript is split into meaningful chunks to help the summarizer focus on sections of the video.

4. **AI Summarization & Highlight Generation**
   A large language model (e.g., Gemini) analyzes the transcript chunks to:

   * Create a **concise summary**.
   * Identify **key phrases and highlights** in the video.

5. **Visual Analysis (Optional)**

   * **Frame captions** generated with BLIP.
   * **OCR** extracts on-screen text (e.g., slides).
   * Object and scene recognition via OpenCV to add context.

6. **Output Display**
   Streamlit displays the full transcript, summary, and any visual insights.

---

## 🛠️ Tech Stack

| Category             | Technology / Library  | Purpose                               |
| -------------------- | --------------------- | ------------------------------------- |
| **Frontend**         | Streamlit             | UI for file upload and result viewing |
| **Transcription**    | Whisper (OpenAI)      | Converts audio to text                |
| **Summarization**    | Gemini / LLM (OpenAI) | Generates summaries & highlights      |
| **Video Processing** | MoviePy               | Extracts audio and frames             |
| **Captioning**       | BLIP                  | Generates visual captions             |
| **OCR**              | EasyOCR               | Detects text from video frames        |
| **Computer Vision**  | OpenCV                | Helps extract additional context      |
| **Core Language**    | Python                | Main application language             |

---

## 🚀 Getting Started

Follow these instructions to set up and run the project locally.

### Prerequisites

* Python 3.9 or higher
* `pip` and a virtual environment for dependency management

### 1. Clone the Repository

```bash
git clone https://github.com/kanhaiyasoni08/Ai-video-summarizer.git
cd Ai-video-summarizer
```

---

### 2. Set Up a Virtual Environment

```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Running the Streamlit App

```bash
streamlit run main.py
```

This command will open the Streamlit UI in your browser, usually at `http://localhost:8501`.

---

## 🧠 Usage Guide

1. **Upload Video:** Click the upload button and select a video file.
2. **Transcribe:** The app auto-extracts audio and runs Whisper to transcribe.
3. **Summarize:** The LLM summarizes the transcript for key insights.
4. **View Results:** See the full transcript, summary text, and optional visual annotations directly in the UI.

---

## 🧩 Example Workflow

1. You upload a 30-minute lecture video.
2. Whisper generates the transcript.
3. The transcript is split into chunks.
4. LLM summarizes each chunk and produces a final overview.
5. Visual captioning/ OCR highlights slide text and important visuals.
6. Results are displayed and ready for download *or export*.

---

## 📈 Future Improvements

Here are some suggested enhancements:

* **YouTube URL Input:** Accept YouTube links directly and auto-download videos.
* **Export Options:** Generate `.txt`, `.PDF`, or `.DOCX` summary downloads.
* **Realtime Processing:** Add progress indicators for lengthy videos.
* **Cloud Deployment:** Deploy on cloud platforms like Heroku, Render, or Hugging Face Spaces.
* **Batch Processing:** Allow uploading multiple videos and generating combined summaries.
