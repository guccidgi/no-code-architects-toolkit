#!/usr/bin/env python
import os
import sys

os.environ["WHISPER_CACHE_DIR"] = "/app/whisper_cache"
os.environ["XDG_CACHE_HOME"] = "/app/cache"
print("Using cache dir:", os.environ.get("WHISPER_CACHE_DIR"))

try:
    import whisper
    print("Checking for Whisper model...")
    if not os.path.exists(os.path.join(os.environ["WHISPER_CACHE_DIR"], "base.pt")):
        print("Downloading Whisper base model...")
        whisper.load_model("base")
        print("Whisper model downloaded successfully")
    else:
        print("Whisper model already exists")
except Exception as e:
    print(f"Error loading Whisper model: {e}")
    # Continue even if model download fails
