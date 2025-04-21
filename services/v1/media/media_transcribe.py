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
from datetime import timedelta
from whisper.utils import WriteSRT, WriteVTT
from services.file_management import download_file
import logging
from config import LOCAL_STORAGE_PATH

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def is_english_word(word):
    """Check if a word is an English word (contains only Latin characters, numbers, and common punctuation)."""
    # 英文單詞通常由拉丁字母、數字和常見標點符號組成
    return all(ord(c) < 128 for c in word)


def identify_words(text):
    """Identify words in text, treating English words as units and Chinese characters individually."""
    # 先按空格分割
    parts = text.split()
    words = []
    
    # 嘗試導入 jieba 庫用於中文分詞
    try:
        import jieba
        JIEBA_AVAILABLE = True
        logger.info("Jieba 分詞庫已載入，將用於智能中文詞語分割")
    except ImportError:
        JIEBA_AVAILABLE = False
        logger.info("Jieba 分詞庫未安裝，將使用基本中文字符分割方法")
    
    for part in parts:
        # 判斷是否為英文單詞
        if is_english_word(part):
            # 英文單詞保持完整
            words.append(part)
        else:
            # 檢查是否為中文字符
            has_chinese = any(u'一' <= c <= u'鿿' for c in part)
            
            if has_chinese:
                # 使用 jieba 進行智能分詞，如果可用
                if JIEBA_AVAILABLE:
                    # 先將英文單詞與中文字符分開處理
                    # 使用更安全的方法，不在最終輸出中引入特殊符號
                    
                    # 使用 jieba 對原始文本進行分詞，保持原文順序
                    # 創建一個字符索引到分詞結果的映射
                    char_to_word_map = {}
                    
                    # 使用 jieba 對原始文本進行分詞
                    seg_list = list(jieba.cut(part))
                    
                    # 建立字符索引到分詞的映射
                    char_index = 0
                    for seg in seg_list:
                        seg_len = len(seg)
                        for i in range(seg_len):
                            char_to_word_map[char_index + i] = seg
                        char_index += seg_len
                    
                    # 逐字符處理原始文本，保持原文順序
                    i = 0
                    processed_words = []
                    while i < len(part):
                        # 英文字符直接保持完整
                        if is_english_word(part[i]):
                            current_eng = ''
                            while i < len(part) and is_english_word(part[i]):
                                current_eng += part[i]
                                i += 1
                            if current_eng.strip():
                                processed_words.append(current_eng.strip())
                        # 中文字符使用 jieba 分詞結果
                        else:
                            if i < len(part) and i in char_to_word_map:
                                word = char_to_word_map[i]
                                if word.strip() and word not in processed_words[-1:] if processed_words else True:
                                    processed_words.append(word.strip())
                                i += len(word)
                            else:
                                # 如果沒有對應的分詞結果，則单獨處理該字符
                                if part[i].strip():
                                    processed_words.append(part[i].strip())
                                i += 1
                    
                    # 將處理後的詞語添加到結果中
                    for word in processed_words:
                        if word.strip():  # 確保不添加空字符串
                            words.append(word.strip())
                else:
                    # jieba 不可用，使用基本的字符處理方法
                    # 可以固定「化身」這個常見詞組
                    if '化身' in part:
                        # 特別處理化身這個關鍵詞組
                        start_idx = part.find('化身')
                        end_idx = start_idx + 2  # '化身'為2個字符
                        
                        # 處理前綴
                        if start_idx > 0:
                            for char in part[:start_idx]:
                                words.append(char)
                                
                        # 將化身作為一個整體
                        words.append('化身')
                        
                        # 處理後綴
                        if end_idx < len(part):
                            for char in part[end_idx:]:
                                words.append(char)
                    else:
                        # 一般情況，逐字符處理
                        current_english = ''
                        
                        for char in part:
                            # 判斷是否為英文字符
                            if is_english_word(char):
                                # 累積英文字符
                                current_english += char
                            else:
                                # 如果已經有累積的英文字符，先添加到結果中
                                if current_english:
                                    words.append(current_english)
                                    current_english = ''
                                
                                # 中文字符單獨處理，確保逐字高亮
                                words.append(char)
                        
                        # 添加最後累積的英文字符
                        if current_english:
                            words.append(current_english)
            else:
                # 非中文非英文的字符序列，保持完整
                words.append(part)
    
    return words

def process_transcribe_media(media_url, task, include_text, include_srt, include_segments, word_timestamps, response_type, language, job_id, words_per_line=None):
    """Transcribe or translate media and return the transcript/translation, SRT or VTT file path."""
    logger.info(f"Starting {task} for media URL: {media_url}")
    input_filename = download_file(media_url, os.path.join(LOCAL_STORAGE_PATH, f"{job_id}_input"))
    logger.info(f"Downloaded media to local file: {input_filename}")

    try:
        # u5c0eu5165u9700u8981u7684u5eab
        import subprocess
        import json
        import tempfile
        from pathlib import Path
        
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
            
        # u6aa2u67e5u662fu5426u9700u8981u5206u6bb5u8655u7406uff08u91ddu5c0du8f03u9577u7684u5f71u7247uff09
        try:
            # u4f7fu7528 ffprobe u7372u53d6u5f71u7247u9577u5ea6
            probe_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", input_filename]
            probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
            duration_info = json.loads(probe_result.stdout)
            video_duration = float(duration_info['format']['duration'])
            
            # u5982u679cu5f71u7247u9577u5ea6u8d85u904e 10 u5206u9418uff0cu4f7fu7528u5206u6bb5u8655u7406
            SEGMENT_THRESHOLD = 10 * 60  # 10 u5206u9418u95beu503c
            SEGMENT_LENGTH = 5 * 60      # 5 u5206u9418u6bb5u9577
            
            if video_duration > SEGMENT_THRESHOLD:
                logger.info(f"Video duration is {video_duration} seconds, using segmented processing")
                
                # u6e96u5099u7d50u679cu5bb9u5668
                all_text = ""
                all_segments = []
                all_srt_subtitles = []
                subtitle_index = 1
                
                # u5275u5efau81e8u6642u76eeu9304u5b58u653eu5206u6bb5
                with tempfile.TemporaryDirectory() as temp_dir:
                    segment_count = int(video_duration / SEGMENT_LENGTH) + 1
                    
                    for i in range(segment_count):
                        start_time = i * SEGMENT_LENGTH
                        # u78bau4fddu6700u5f8cu4e00u6bb5u4e0du8d85u51fau5f71u7247u9577u5ea6
                        duration = min(SEGMENT_LENGTH, video_duration - start_time)
                        
                        if duration <= 0:
                            break
                            
                        # u5206u6bb5u6587u4ef6u8defu5f91
                        segment_path = Path(temp_dir) / f"segment_{i}.mp4"
                        
                        # u4f7fu7528 ffmpeg u5207u5272u5f71u7247
                        segment_cmd = [
                            "ffmpeg", "-y", "-i", input_filename,
                            "-ss", str(start_time), "-t", str(duration),
                            "-c", "copy", str(segment_path)
                        ]
                        
                        logger.info(f"Extracting segment {i+1}/{segment_count} from {start_time} to {start_time+duration}")
                        subprocess.run(segment_cmd, capture_output=True)
                        
                        # u8655u7406u5206u6bb5
                        logger.info(f"Processing segment {i+1}/{segment_count}")
                        segment_result = model.transcribe(str(segment_path), **options)
                        
                        # u7d2fu7a4du6587u672c
                        if include_text:
                            all_text += segment_result['text'] + " "
                        
                        # u8abfu6574u6642u9593u6233u4e26u7d2fu7a4du5206u6bb5
                        if include_segments or include_srt:
                            for segment in segment_result['segments']:
                                # u8abfu6574u6642u9593u6233u4ee5u5339u914du539fu59cbu5f71u7247
                                segment['start'] += start_time
                                segment['end'] += start_time
                                
                                if include_segments:
                                    all_segments.append(segment)
                                
                                if include_srt:
                                    if words_per_line and words_per_line > 0:
                                        # u8655u7406u55aeu8a5eu6642u9593u6233
                                        words = identify_words(segment['text'].strip())
                                        segment_start = segment['start']
                                        segment_end = segment['end']
                                        
                                        if words:
                                            duration_per_word = (segment_end - segment_start) / len(words)
                                            word_timings = []
                                            
                                            for j, word in enumerate(words):
                                                word_start = segment_start + (j * duration_per_word)
                                                word_end = word_start + duration_per_word
                                                word_timings.append((word, word_start, word_end))
                                            
                                            # u6309u6bcfu884cu5b57u6578u5206u7d44
                                            for j in range(0, len(word_timings), words_per_line):
                                                chunk = word_timings[j:j+words_per_line]
                                                if chunk:
                                                    chunk_start = chunk[0][1]
                                                    chunk_end = chunk[-1][2]
                                                    chunk_text = ' '.join(w[0] for w in chunk)
                                                    
                                                    all_srt_subtitles.append(srt.Subtitle(
                                                        subtitle_index,
                                                        timedelta(seconds=chunk_start),
                                                        timedelta(seconds=chunk_end),
                                                        chunk_text
                                                    ))
                                                    subtitle_index += 1
                                    else:
                                        # u6bcfu500bu5206u6bb5u4e00u500bu5b57u5e55
                                        all_srt_subtitles.append(srt.Subtitle(
                                            subtitle_index,
                                            timedelta(seconds=segment['start']),
                                            timedelta(seconds=segment['end']),
                                            segment['text'].strip()
                                        ))
                                        subtitle_index += 1
                
                # u6574u7406u6700u7d42u7d50u679c
                result = {'text': all_text.strip(), 'segments': all_segments}
                
                if include_srt:
                    srt_text = srt.compose(all_srt_subtitles)
                else:
                    srt_text = None
                    
                logger.info(f"Completed segmented processing of {segment_count} segments")
            else:
                # u5c0du65bcu8f03u77edu7684u5f71u7247uff0cu4f7fu7528u539fu59cbu8655u7406u65b9u5f0f
                logger.info(f"Video duration is {video_duration} seconds, using standard processing")
                result = model.transcribe(input_filename, **options)
        except Exception as e:
            # u5982u679cu5206u6bb5u8655u7406u5931u6557uff0cu56deu9000u5230u6a19u6e96u8655u7406
            logger.warning(f"Segmented processing failed: {str(e)}. Falling back to standard processing.")
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
                # Collect all words and their timings
                all_words = []
                word_timings = []
                
                for segment in result['segments']:
                    # 使用 identify_words 函數來處理中文和英文混合文本
                    words = identify_words(segment['text'].strip())
                    segment_start = segment['start']
                    segment_end = segment['end']
                    
                    # Calculate timing for each word
                    if words:
                        duration_per_word = (segment_end - segment_start) / len(words)
                        for i, word in enumerate(words):
                            word_start = segment_start + (i * duration_per_word)
                            word_end = word_start + duration_per_word
                            all_words.append(word)
                            word_timings.append((word_start, word_end))
                
                # Process words in chunks of words_per_line
                current_word = 0
                while current_word < len(all_words):
                    # Get the next chunk of words
                    chunk = all_words[current_word:current_word + words_per_line]
                    
                    # Calculate timing for this chunk
                    chunk_start = word_timings[current_word][0]
                    chunk_end = word_timings[min(current_word + len(chunk) - 1, len(word_timings) - 1)][1]
                    
                    # Create the subtitle
                    srt_subtitles.append(srt.Subtitle(
                        subtitle_index,
                        timedelta(seconds=chunk_start),
                        timedelta(seconds=chunk_end),
                        ' '.join(chunk)
                    ))
                    subtitle_index += 1
                    current_word += words_per_line
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