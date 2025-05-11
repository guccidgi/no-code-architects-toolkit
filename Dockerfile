# Base image
FROM python:3.9-slim

# Install system dependencies, build tools, and libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    wget \
    tar \
    xz-utils \
    fonts-liberation \
    fontconfig \
    build-essential \
    yasm \
    cmake \
    meson \
    ninja-build \
    nasm \
    libssl-dev \
    libvpx-dev \
    libx264-dev \
    libx265-dev \
    libnuma-dev \
    libmp3lame-dev \
    libopus-dev \
    libvorbis-dev \
    libtheora-dev \
    libspeex-dev \
    libfreetype6-dev \
    libfontconfig1-dev \
    libgnutls28-dev \
    libaom-dev \
    libdav1d-dev \
    librav1e-dev \
    libsvtav1-dev \
    libzimg-dev \
    libwebp-dev \
    git \
    pkg-config \
    autoconf \
    automake \
    libtool \
    libfribidi-dev \
    libharfbuzz-dev \
    && rm -rf /var/lib/apt/lists/*

# Install SRT from source (latest version using cmake)
RUN git clone https://github.com/Haivision/srt.git && \
    cd srt && \
    mkdir build && cd build && \
    cmake .. && \
    make -j$(nproc) && \
    make install && \
    cd ../.. && rm -rf srt

# Install SVT-AV1 from source
RUN git clone https://gitlab.com/AOMediaCodec/SVT-AV1.git && \
    cd SVT-AV1 && \
    git checkout v0.9.0 && \
    cd Build && \
    cmake .. && \
    make -j$(nproc) && \
    make install && \
    cd ../.. && rm -rf SVT-AV1

# Install libvmaf from source
RUN git clone https://github.com/Netflix/vmaf.git && \
    cd vmaf/libvmaf && \
    meson build --buildtype release && \
    ninja -C build && \
    ninja -C build install && \
    cd ../.. && rm -rf vmaf && \
    ldconfig  # Update the dynamic linker cache

# Manually build and install fdk-aac (since it is not available via apt-get)
RUN git clone https://github.com/mstorsjo/fdk-aac && \
    cd fdk-aac && \
    autoreconf -fiv && \
    ./configure && \
    make -j$(nproc) && \
    make install && \
    cd .. && rm -rf fdk-aac

# Install libunibreak (required for ASS_FEATURE_WRAP_UNICODE)
RUN git clone https://github.com/adah1972/libunibreak.git && \
    cd libunibreak && \
    ./autogen.sh && \
    ./configure && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    cd .. && rm -rf libunibreak

# Build and install libass with libunibreak support and ASS_FEATURE_WRAP_UNICODE enabled
RUN git clone https://github.com/libass/libass.git && \
    cd libass && \
    autoreconf -i && \
    ./configure --enable-libunibreak || { cat config.log; exit 1; } && \
    mkdir -p /app && echo "Config log located at: /app/config.log" && cp config.log /app/config.log && \
    make -j$(nproc) || { echo "Libass build failed"; exit 1; } && \
    make install && \
    ldconfig && \
    cd .. && rm -rf libass

# Build and install FFmpeg with all required features
RUN git clone https://git.ffmpeg.org/ffmpeg.git ffmpeg && \
    cd ffmpeg && \
    git checkout n7.0.2 && \
    PKG_CONFIG_PATH="/usr/lib/x86_64-linux-gnu/pkgconfig:/usr/local/lib/pkgconfig" \
    CFLAGS="-I/usr/include/freetype2" \
    LDFLAGS="-L/usr/lib/x86_64-linux-gnu" \
    ./configure --prefix=/usr/local \
        --enable-gpl \
        --enable-pthreads \
        --enable-neon \
        --enable-libaom \
        --enable-libdav1d \
        --enable-librav1e \
        --enable-libsvtav1 \
        --enable-libvmaf \
        --enable-libzimg \
        --enable-libx264 \
        --enable-libx265 \
        --enable-libvpx \
        --enable-libwebp \
        --enable-libmp3lame \
        --enable-libopus \
        --enable-libvorbis \
        --enable-libtheora \
        --enable-libspeex \
        --enable-libass \
        --enable-libfreetype \
        --enable-libharfbuzz \
        --enable-fontconfig \
        --enable-libsrt \
        --enable-filter=drawtext \
        --extra-cflags="-I/usr/include/freetype2 -I/usr/include/libpng16 -I/usr/include" \
        --extra-ldflags="-L/usr/lib/x86_64-linux-gnu -lfreetype -lfontconfig" \
        --enable-gnutls \
    && make -j$(nproc) && \
    make install && \
    cd .. && rm -rf ffmpeg

# Add /usr/local/bin to PATH (if not already included)
ENV PATH="/usr/local/bin:${PATH}"

# Copy fonts into the custom fonts directory
COPY ./fonts /usr/share/fonts/custom

# Rebuild the font cache so that fontconfig can see the custom fonts
RUN fc-cache -f -v

# Set work directory
WORKDIR /app

# Set environment variable for Whisper cache
ENV WHISPER_CACHE_DIR="/app/whisper_cache"
ENV XDG_CACHE_HOME="/app/cache"

# Create cache directories with proper permissions
RUN mkdir -p ${WHISPER_CACHE_DIR} && mkdir -p ${XDG_CACHE_HOME}

# Copy the requirements file first to optimize caching
COPY requirements.txt .

# Install Python dependencies, upgrade pip 
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install openai-whisper && \
    pip install jsonschema 

# Create the appuser 
RUN useradd -m appuser 

# Give appuser ownership of the /app directory and all cache directories
RUN chown -R appuser:appuser /app

# Copy the script files and set permissions
COPY run_gunicorn.sh /app/run_gunicorn.sh
COPY download_whisper_model.py /app/download_whisper_model.py
COPY startup.sh /app/startup.sh

# Copy the rest of the application code
COPY . .

# Make the scripts executable and set ownership
RUN chmod -R +x /app/*.sh /app/*.py && \
    chown -R appuser:appuser /app

# Switch to the appuser for security
USER appuser

# Expose the port the app runs on
EXPOSE 8080

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the startup script with bash explicitly
CMD ["/bin/bash", "/app/startup.sh"]