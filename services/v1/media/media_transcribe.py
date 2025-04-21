# Copyright (c) 2025 Stephen G. Pope
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.



import os
import whisper
import srt
import re
from datetime import timedelta
from whisper.utils import WriteSRT, WriteVTT
from services.file_management import download_file
from services.v1.video.caption_video import identify_words, is_english_word
import logging
from config import LOCAL_STORAGE_PATH

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def process_transcribe_media(media_url, task, include_text, include_srt, include_segments, word_timestamps, response_type, language, job_id, words_per_line=None):
    """Transcribe or translate media and return the transcript/translation, SRT or VTT file path."""
    logger.info(f"Starting {task} for media URL: {media_url}")
    input_filename = download_file(media_url, os.path.join(LOCAL_STORAGE_PATH, f"{job_id}_input"))
    logger.info(f"Downloaded media to local file: {input_filename}")

    try:
        # Load a larger model for better translation quality
        #model_size = "large" if task == "translate" else "base"
        model_size = "base"
        model = whisper.load_model(model_size)
        logger.info(f"Loaded Whisper {model_size} model")

        # Configure transcription/translation options
        options = {
            "task": task,
            "word_timestamps": word_timestamps,
            "verbose": False
        }

        # Add language specification if provided
        if language:
            options["language"] = language

        result = model.transcribe(input_filename, **options)
        
        # For translation task, the result['text'] will be in English
        text = None
        srt_text = None
        segments_json = None

        logger.info(f"Generated {task} output")

        if include_text is True:
            text = result['text']

        if include_srt is True:
            srt_subtitles = []
            subtitle_index = 1
            
            if words_per_line and words_per_line > 0:
                # 自行實現中文字符分割，確保正確計算字數
                def split_chinese_text(text):
                    # 使用正則表達式分割中英文
                    # 英文單詞、數字和標點符號作為整體
                    # 中文字符單獨處理
                    pattern = r'[A-Za-z0-9]+|[\u4e00-\u9fff]|[^\s\w\u4e00-\u9fff]+'
                    return [m for m in re.findall(pattern, text) if m.strip()]
                
                # 收集所有文字和時間
                all_words = []
                all_timings = []
                
                # 記錄處理過程
                logger.info(f"Processing with words_per_line={words_per_line}")
                
                # 收集所有文字和對應的時間點
                for segment in result['segments']:
                    seg_text = segment['text'].strip()
                    # 使用我們自己的分詞函數，確保中文字符被正確分割
                    seg_words = split_chinese_text(seg_text)
                    seg_start = segment['start']
                    seg_end = segment['end']
                    
                    if not seg_words:
                        continue
                    
                    logger.info(f"Segment: {seg_text}")
                    logger.info(f"Words count: {len(seg_words)}, Words: {seg_words}")
                    
                    # 計算每個詞的時間
                    duration_per_word = (seg_end - seg_start) / len(seg_words)
                    
                    # 為每個詞計算時間點
                    for i, word in enumerate(seg_words):
                        word_start = seg_start + i * duration_per_word
                        word_end = word_start + duration_per_word
                        all_words.append(word)
                        all_timings.append((word_start, word_end))
                
                logger.info(f"Total words: {len(all_words)}")
                
                # 嚴格按照 words_per_line 分割所有文字
                for i in range(0, len(all_words), words_per_line):
                    chunk = all_words[i:i + words_per_line]
                    
                    if not chunk:
                        continue
                    
                    # 計算這個 chunk 的開始和結束時間
                    chunk_start = all_timings[i][0]
                    chunk_end = all_timings[min(i + len(chunk) - 1, len(all_timings) - 1)][1]
                    
                    # 檢查是否全為中文
                    has_chinese = False
                    has_non_chinese = False
                    
                    for word in chunk:
                        if any('\u4e00' <= c <= '\u9fff' for c in word):
                            has_chinese = True
                        else:
                            has_non_chinese = True
                    
                    # 如果全是中文，不加空格；否則用空格連接
                    if has_chinese and not has_non_chinese:
                        subtitle_text = ''.join(chunk)
                    else:
                        subtitle_text = ' '.join(chunk)
                    
                    logger.info(f"Subtitle {subtitle_index}: {subtitle_text} (words: {len(chunk)})")
                    
                    # 建立字幕
                    srt_subtitles.append(srt.Subtitle(
                        subtitle_index,
                        timedelta(seconds=chunk_start),
                        timedelta(seconds=chunk_end),
                        subtitle_text
                    ))
                    subtitle_index += 1
            else:
                # Original behavior - one subtitle per segment
                for segment in result['segments']:
                    start = timedelta(seconds=segment['start'])
                    end = timedelta(seconds=segment['end'])
                    segment_text = segment['text'].strip()
                    srt_subtitles.append(srt.Subtitle(subtitle_index, start, end, segment_text))
                    subtitle_index += 1
            
            srt_text = srt.compose(srt_subtitles)

        if include_segments is True:
            segments_json = result['segments']

        os.remove(input_filename)
        logger.info(f"Removed local file: {input_filename}")
        logger.info(f"{task.capitalize()} successful, output type: {response_type}")

        if response_type == "direct":
            return text, srt_text, segments_json
        else:
            
            if include_text is True:
                text_filename = os.path.join(LOCAL_STORAGE_PATH, f"{job_id}.txt")
                with open(text_filename, 'w') as f:
                    f.write(text)
            else:
                text_file = None
            
            if include_srt is True:
                srt_filename = os.path.join(LOCAL_STORAGE_PATH, f"{job_id}.srt")
                with open(srt_filename, 'w') as f:
                    f.write(srt_text)
            else:
                srt_filename = None

            if include_segments is True:
                segments_filename = os.path.join(LOCAL_STORAGE_PATH, f"{job_id}.json")
                with open(segments_filename, 'w') as f:
                    f.write(str(segments_json))
            else:
                segments_filename = None

            return text_filename, srt_filename, segments_filename 

    except Exception as e:
        logger.error(f"{task.capitalize()} failed: {str(e)}")
        raise