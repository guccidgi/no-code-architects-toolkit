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
import jieba

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
                # 使用 jieba 進行中文分詞，確保完整詞彙不被分割
                def split_text_with_jieba(text):
                    # 檢測文本中是否包含中文
                    has_chinese = any('\u4e00' <= c <= '\u9fff' for c in text)
                    
                    if has_chinese:
                        # 使用 jieba 進行中文分詞
                        jieba_words = list(jieba.cut(text))
                        # 進一步處理分詞結果，處理混合中英文的情況
                        words = []
                        for word in jieba_words:
                            # 如果是英文單詞，保持完整
                            if is_english_word(word):
                                words.append(word)
                            else:
                                # 對於中文詞彙或其他字符，使用 identify_words 進一步處理
                                words.extend(identify_words(word))
                        return [w for w in words if w.strip()]
                    else:
                        # 純英文文本，使用空格分割
                        return [w for w in text.split() if w.strip()]
                
                # 收集所有文字和時間
                all_words = []
                all_timings = []
                all_segments = []
                
                # 記錄處理過程
                logger.info(f"Processing with words_per_line={words_per_line}")
                
                # 收集所有文字和對應的時間點
                for segment in result['segments']:
                    seg_text = segment['text'].strip()
                    # 使用 jieba 進行分詞
                    seg_words = split_text_with_jieba(seg_text)
                    seg_start = segment['start']
                    seg_end = segment['end']
                    
                    if not seg_words:
                        continue
                    
                    logger.info(f"Segment: {seg_text}")
                    logger.info(f"Words count: {len(seg_words)}, Words: {seg_words}")
                    
                    # 保存段落信息，用於後續智能分割
                    all_segments.append({
                        'words': seg_words,
                        'start': seg_start,
                        'end': seg_end
                    })
                    
                    # 計算每個詞的時間
                    duration_per_word = (seg_end - seg_start) / len(seg_words)
                    
                    # 為每個詞計算時間點
                    for i, word in enumerate(seg_words):
                        word_start = seg_start + i * duration_per_word
                        word_end = word_start + duration_per_word
                        all_words.append(word)
                        all_timings.append((word_start, word_end))
                
                logger.info(f"Total words: {len(all_words)}")
                
                # 智能分割字幕，確保完整詞彙不被分割，同時嚴格遵守每行最大字數
                def smart_chunk_words(words, timings, target_words_per_line):
                    chunks = []
                    current_chunk = []
                    current_timings = []
                    current_count = 0
                    
                    i = 0
                    while i < len(words):
                        word = words[i]
                        timing = timings[i]
                        
                        # 計算當前詞的權重
                        # 對於中文字符，每個字計為1；對於英文單詞，整個單詞計為1
                        if any('\u4e00' <= c <= '\u9fff' for c in word):
                            # 中文詞彙，按字符數計算權重
                            chinese_chars = sum(1 for c in word if '\u4e00' <= c <= '\u9fff')
                            non_chinese_chars = len(word) - chinese_chars
                            word_weight = chinese_chars + (1 if non_chinese_chars > 0 else 0)
                        else:
                            # 英文單詞或其他，整體計為1
                            word_weight = 1
                        
                        # 如果添加這個詞會超過目標行數，且當前行已有內容，則結束當前行
                        if current_count + word_weight > target_words_per_line and current_chunk:
                            chunks.append((current_chunk, current_timings))
                            current_chunk = []
                            current_timings = []
                            current_count = 0
                        
                        # 如果單個詞的權重已經超過目標行數，且這是一個較長的詞，考慮拆分
                        # 但僅適用於中文詞彙，英文詞彙保持完整
                        if word_weight > target_words_per_line and any('\u4e00' <= c <= '\u9fff' for c in word):
                            # 將長中文詞彙拆分為多個小塊
                            chars = list(word)
                            for j in range(0, len(chars), target_words_per_line):
                                sub_word = ''.join(chars[j:j+target_words_per_line])
                                if not sub_word:
                                    continue
                                
                                # 為拆分的詞計算時間比例
                                sub_duration = (timing[1] - timing[0]) * (len(sub_word) / len(word))
                                sub_start = timing[0] + (timing[1] - timing[0]) * (j / len(word))
                                sub_end = sub_start + sub_duration
                                
                                # 如果當前行已有內容且添加會超過限制，先結束當前行
                                if current_count + len(sub_word) > target_words_per_line and current_chunk:
                                    chunks.append((current_chunk, current_timings))
                                    current_chunk = []
                                    current_timings = []
                                    current_count = 0
                                
                                current_chunk.append(sub_word)
                                current_timings.append((sub_start, sub_end))
                                current_count += len(sub_word)
                                
                                # 如果當前行已滿，結束當前行
                                if current_count >= target_words_per_line:
                                    chunks.append((current_chunk, current_timings))
                                    current_chunk = []
                                    current_timings = []
                                    current_count = 0
                            i += 1
                            continue
                        
                        # 添加詞到當前行
                        current_chunk.append(word)
                        current_timings.append(timing)
                        current_count += word_weight
                        i += 1
                        
                        # 檢查是否需要結束當前行（句號、問號、驚嘆號等）
                        if i < len(words) and current_count >= target_words_per_line * 0.5:
                            next_few_words = ''.join(words[i:i+3]) if i+3 <= len(words) else ''.join(words[i:])
                            if any(p in word for p in ['。', '.', '!', '?', '！', '？', '…', '，', ',', '；', ';']) or \
                               any(p in next_few_words for p in ['。', '.', '!', '?', '！', '？', '…']):
                                chunks.append((current_chunk, current_timings))
                                current_chunk = []
                                current_timings = []
                                current_count = 0
                    
                    # 添加最後一行（如果有）
                    if current_chunk:
                        chunks.append((current_chunk, current_timings))
                    
                    return chunks
                
                # 使用智能分割算法
                chunks = smart_chunk_words(all_words, all_timings, words_per_line)
                
                # 創建字幕
                for chunk_words, chunk_timings in chunks:
                    if not chunk_words:
                        continue
                    
                    # 計算這個 chunk 的開始和結束時間
                    chunk_start = chunk_timings[0][0]
                    chunk_end = chunk_timings[-1][1]
                    
                    # 檢查是否包含中文
                    has_chinese = any(any('\u4e00' <= c <= '\u9fff' for c in word) for word in chunk_words)
                    has_non_chinese = any(not all('\u4e00' <= c <= '\u9fff' for c in word) for word in chunk_words if word.strip())
                    
                    # 根據內容類型決定如何連接詞彙
                    if has_chinese and not has_non_chinese:
                        # 純中文內容，不加空格
                        subtitle_text = ''.join(chunk_words)
                    else:
                        # 混合內容或純英文，需要智能處理空格
                        subtitle_text = ''
                        for i, word in enumerate(chunk_words):
                            if i > 0:
                                # 如果當前詞和前一個詞都不是中文，添加空格
                                prev_is_chinese = any('\u4e00' <= c <= '\u9fff' for c in chunk_words[i-1])
                                curr_is_chinese = any('\u4e00' <= c <= '\u9fff' for c in word)
                                
                                if not (prev_is_chinese or curr_is_chinese):
                                    subtitle_text += ' '
                            subtitle_text += word
                    
                    logger.info(f"Subtitle {subtitle_index}: {subtitle_text} (words: {len(chunk_words)})")
                    
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