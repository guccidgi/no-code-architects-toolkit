#!/bin/bash

echo "=== Starting container debugging ==="

# Display environment
echo "Environment variables:"
env

# Check file permissions
echo "
File permissions in /app:"
ls -la /app

# Check Python installation
echo "
Python version:"
python --version

# Check if we can import Flask
echo "
Testing Flask import:"
python -c "import flask; print('Flask version:', flask.__version__)" || echo "Failed to import Flask"

# Try to download Whisper model but continue if it fails
echo "
Attempting to download Whisper model:"
python /app/download_whisper_model.py || echo "Failed to download Whisper model - continuing anyway"

# Try to import app module
echo "
Testing app module:"
python -c "import app; print('App import successful')" || echo "Failed to import app module"

echo "
=== Starting gunicorn server ==="

# Start the gunicorn server with debug logging
exec gunicorn --bind 0.0.0.0:8080 \
    --workers ${GUNICORN_WORKERS:-2} \
    --timeout ${GUNICORN_TIMEOUT:-300} \
    --worker-class sync \
    --keep-alive 80 \
    --log-level debug \
    app:app
