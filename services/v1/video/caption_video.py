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
import ffmpeg
import logging
import subprocess
import whisper
from datetime import timedelta
import srt
import re
from services.file_management import download_file
from services.cloud_storage import upload_file  # Ensure this import is present
import requests  # Ensure requests is imported for webhook handling
from urllib.parse import urlparse
from config import LOCAL_STORAGE_PATH

# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

POSITION_ALIGNMENT_MAP = {
    "bottom_left": 1,
    "bottom_center": 2,
    "bottom_right": 3,
    "middle_left": 4,
    "middle_center": 5,
    "middle_right": 6,
    "top_left": 7,
    "top_center": 8,
    "top_right": 9
}

def rgb_to_ass_color(rgb_color):
    """Convert RGB hex to ASS (&HAABBGGRR)."""
    if isinstance(rgb_color, str):
        rgb_color = rgb_color.lstrip('#')
        if len(rgb_color) == 6:
            r = int(rgb_color[0:2], 16)
            g = int(rgb_color[2:4], 16)
            b = int(rgb_color[4:6], 16)
            return f"&H00{b:02X}{g:02X}{r:02X}"
    return "&H00FFFFFF"

def generate_transcription(video_path, language='auto'):
    try:
        model = whisper.load_model("base")
        transcription_options = {
            'word_timestamps': True,
            'verbose': True,
        }
        if language != 'auto':
            transcription_options['language'] = language
        result = model.transcribe(video_path, **transcription_options)
        logger.info(f"Transcription generated successfully for video: {video_path}")
        return result
    except Exception as e:
        logger.error(f"Error in transcription: {str(e)}")
        raise

def get_video_resolution(video_path):
    try:
        probe = ffmpeg.probe(video_path)
        video_streams = [s for s in probe['streams'] if s['codec_type'] == 'video']
        if video_streams:
            width = int(video_streams[0]['width'])
            height = int(video_streams[0]['height'])
            logger.info(f"Video resolution determined: {width}x{height}")
            return width, height
        else:
            logger.warning(f"No video streams found for {video_path}. Using default resolution 384x288.")
            return 384, 288
    except Exception as e:
        logger.error(f"Error getting video resolution: {str(e)}. Using default resolution 384x288.")
        return 384, 288

def get_available_fonts():
    """Get the list of available fonts on the system."""
    try:
        import matplotlib.font_manager as fm
    except ImportError:
        logger.error("matplotlib not installed. Install via 'pip install matplotlib'.")
        return []
    font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')
    font_names = set()
    for font in font_list:
        try:
            font_prop = fm.FontProperties(fname=font)
            font_name = font_prop.get_name()
            font_names.add(font_name)
        except Exception:
            continue
    logger.info(f"Available fonts retrieved: {font_names}")
    return list(font_names)

def format_ass_time(seconds):
    """Convert float seconds to ASS time format H:MM:SS.cc"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centiseconds = int(round((seconds - int(seconds)) * 100))
    return f"{hours}:{minutes:02}:{secs:02}.{centiseconds:02}"

def process_subtitle_text(text, replace_dict, all_caps, max_words_per_line):
    """Apply text transformations: replacements, all caps, and optional line splitting."""
    for old_word, new_word in replace_dict.items():
        text = re.sub(re.escape(old_word), new_word, text, flags=re.IGNORECASE)
    if all_caps:
        text = text.upper()
    if max_words_per_line > 0:
        # 使用 identify_words 函數而不是簡單的 text.split()
        words = identify_words(text)
        
        # 使用 identify_words 函數來智能識別詞語邊界，替代硬編碼的詞組模式
        
        # 注意：我們不再需要手動合併詞組
        # 因為現在使用 identify_words 函數進行智能分詞
        # 如果使用 jieba，它會自動識別詞語邊界
        # 如果沒有 jieba，我們在 identify_words 中已經處理「化身」等常見詞組
        
        # 智能分行處理，考慮中文詞語的完整性
        lines = []
        current_line = []
        word_count = 0
        
        # 檢查是否有中文字符
        has_chinese = any(u'一' <= c <= u'鿿' for c in text)
        
        # 如果是中文文本，使用更智能的分行邏輯
        if has_chinese:
            # 獲取詞語邊界，確保不會在詞中間斷開
            # 這裡我們使用 identify_words 函數來獲取詞語邊界
            word_groups = []
            current_group = []
            
            # 識別中文詞語組（例如「化身」應該被視為一個整體）
            i = 0
            while i < len(words):
                current_word = words[i]
                # 檢查是否為中文字符
                is_chinese = any(u'一' <= c <= u'鿿' for c in current_word)
                
                # 如果是中文字符，檢查下一個是否也是中文並可能形成詞語
                if is_chinese and i + 1 < len(words):
                    next_word = words[i + 1]
                    next_is_chinese = any(u'一' <= c <= u'鿿' for c in next_word)
                    
                    # 如果當前和下一個都是中文，可能是一個詞語
                    if next_is_chinese and len(current_word + next_word) <= 4:  # 大多數中文詞語不超過4個字
                        # 將這兩個字視為一個詞語組
                        word_groups.append([current_word, next_word])
                        i += 2  # 跳過下一個字
                        continue
                
                # 單個詞
                word_groups.append([current_word])
                i += 1
            
            # 根據詞語組進行分行
            current_line = []
            word_count = 0
            
            for group in word_groups:
                # 檢查添加這個詞語組是否會超過每行最大單詞數
                if word_count + len(group) <= max_words_per_line:
                    current_line.extend(group)
                    word_count += len(group)
                else:
                    # 如果當前行已有內容，則完成當前行
                    if current_line:
                        lines.append(' '.join(current_line))
                    
                    # 開始新的一行
                    current_line = group.copy()
                    word_count = len(group)
        else:
            # 非中文文本使用原來的分行邏輯
            for word in words:
                if word_count < max_words_per_line:
                    current_line.append(word)
                    word_count += 1
                else:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                    word_count = 1
        
        # 添加最後一行
        if current_line:
            lines.append(' '.join(current_line))
        
        text = r'\N'.join(lines)
    return text

def srt_to_transcription_result(srt_content):
    """Convert SRT content into a transcription-like structure for uniform processing.
    Enhanced to add word-level timestamps for advanced styling support."""
    subtitles = list(srt.parse(srt_content))
    segments = []
    
    # 嘗試導入 jieba 庫用於中文分詞
    try:
        import jieba
        JIEBA_AVAILABLE = True
        logger.info("Jieba 分詞庫已載入，將用於智能中文詞語分割")
    except ImportError:
        JIEBA_AVAILABLE = False
        logger.info("Jieba 分詞庫未安裝，將使用基本中文字符分割方法")
    
    for sub in subtitles:
        text = sub.content.strip()
        start_time = sub.start.total_seconds()
        end_time = sub.end.total_seconds()
        duration = end_time - start_time
        
        # 使用智能分詞方法
        def split_text_with_jieba(text):
            # 檢測文本中是否包含中文
            has_chinese = any('\u4e00' <= c <= '\u9fff' for c in text)
            
            if has_chinese and JIEBA_AVAILABLE:
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
                # 使用原有的 identify_words 函數
                return identify_words(text)
        
        # 使用改進的分詞方法
        words_list = split_text_with_jieba(text)
        word_count = len(words_list)
        
        # Generate word-level timestamps with weighted durations
        words = []
        if word_count > 0:
            # 計算每個詞的權重，用於更精確的時間分配
            word_weights = []
            total_weight = 0
            
            for word in words_list:
                # 對於中文字符，每個字計為1；對於英文單詞，整個單詞計為1
                if any('\u4e00' <= c <= '\u9fff' for c in word):
                    # 中文詞彙，按字符數計算權重
                    chinese_chars = sum(1 for c in word if '\u4e00' <= c <= '\u9fff')
                    non_chinese_chars = len(word) - chinese_chars
                    word_weight = chinese_chars + (1 if non_chinese_chars > 0 else 0)
                else:
                    # 英文單詞或其他，整體計為1
                    word_weight = 1
                
                word_weights.append(word_weight)
                total_weight += word_weight
            
            # 根據權重分配時間
            current_time = start_time
            for i, word in enumerate(words_list):
                word_duration = (duration * word_weights[i]) / total_weight if total_weight > 0 else 0
                word_start = current_time
                word_end = word_start + word_duration
                words.append({
                    'word': word,
                    'start': word_start,
                    'end': word_end
                })
                current_time = word_end
        
        segments.append({
            'start': start_time,
            'end': end_time,
            'text': text,
            'words': words  # Now includes word-level timestamps with weighted durations
        })
    
    logger.info("Converted SRT content to transcription result with word-level timestamps and improved word segmentation.")
    return {'segments': segments}

def is_english_word(word):
    """Check if a word is an English word (contains only Latin characters, numbers, and common punctuation)."""
    # 英文單詞通常由拉丁字母、數字和常見標點符號組成
    return all(ord(c) < 128 for c in word)

def smart_join_words(words):
    """智能連接字詞，根據中英文特性決定是否添加空格。"""
    if not words:
        return ''
    
    result = ''
    for i, word in enumerate(words):
        if i > 0:
            # 檢查當前詞和前一個詞是否為中文
            prev_is_chinese = any('\u4e00' <= c <= '\u9fff' for c in words[i-1])
            curr_is_chinese = any('\u4e00' <= c <= '\u9fff' for c in word)
            
            # 如果當前詞和前一個詞都不是中文，添加空格
            if not (prev_is_chinese or curr_is_chinese):
                result += ' '
        result += word
    
    return result

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

def split_lines(text, max_words_per_line):
    """Split text into multiple lines if max_words_per_line > 0, preserving word integrity."""
    if max_words_per_line <= 0:
        return [text]
    
    # 使用新的 identify_words 函數識別單詞
    words = identify_words(text)
    
    # 按指定的每行最大單詞數分割
    lines = []
    for i in range(0, len(words), max_words_per_line):
        line = ' '.join(words[i:i+max_words_per_line])
        lines.append(line)
    
    return lines

def is_url(string):
    """Check if the given string is a valid HTTP/HTTPS URL."""
    try:
        result = urlparse(string)
        return result.scheme in ('http', 'https')
    except:
        return False

def download_captions(captions_url):
    """Download captions from the given URL."""
    try:
        logger.info(f"Downloading captions from URL: {captions_url}")
        response = requests.get(captions_url)
        response.raise_for_status()
        logger.info("Captions downloaded successfully.")
        return response.text
    except Exception as e:
        logger.error(f"Error downloading captions: {str(e)}")
        raise

def determine_alignment_code(position_str, alignment_str, x, y, video_width, video_height):
    """
    Determine the final \an alignment code and (x,y) position based on:
    - x,y (if provided)
    - position_str (one of top_left, top_center, ...)
    - alignment_str (left, center, right)
    - If x,y not provided, divide the video into a 3x3 grid and position accordingly.
    """
    logger.info(f"[determine_alignment_code] Inputs: position_str={position_str}, alignment_str={alignment_str}, x={x}, y={y}, video_width={video_width}, video_height={video_height}")

    horizontal_map = {
        'left': 1,
        'center': 2,
        'right': 3
    }

    # If x and y are provided, use them directly and set \an based on alignment_str
    if x is not None and y is not None:
        logger.info("[determine_alignment_code] x and y provided, ignoring position and alignment for grid.")
        vertical_code = 4  # Middle row
        horiz_code = horizontal_map.get(alignment_str, 2)  # Default to center
        an_code = vertical_code + (horiz_code - 1)
        logger.info(f"[determine_alignment_code] Using provided x,y. an_code={an_code}")
        return an_code, True, x, y

    # No x,y provided: determine position and alignment based on grid
    pos_lower = position_str.lower()
    if 'top' in pos_lower:
        vertical_base = 7  # Top row an codes start at 7
        vertical_center = video_height / 6
    elif 'middle' in pos_lower:
        vertical_base = 4  # Middle row an codes start at 4
        vertical_center = video_height / 2
    else:
        vertical_base = 1  # Bottom row an codes start at 1
        vertical_center = (5 * video_height) / 6

    if 'left' in pos_lower:
        left_boundary = 0
        right_boundary = video_width / 3
        center_line = video_width / 6
    elif 'right' in pos_lower:
        left_boundary = (2 * video_width) / 3
        right_boundary = video_width
        center_line = (5 * video_width) / 6
    else:
        # Center column
        left_boundary = video_width / 3
        right_boundary = (2 * video_width) / 3
        center_line = video_width / 2

    # Alignment affects horizontal position within the cell
    if alignment_str == 'left':
        final_x = left_boundary
        horiz_code = 1
    elif alignment_str == 'right':
        final_x = right_boundary
        horiz_code = 3
    else:
        final_x = center_line
        horiz_code = 2

    final_y = vertical_center
    an_code = vertical_base + (horiz_code - 1)

    logger.info(f"[determine_alignment_code] Computed final_x={final_x}, final_y={final_y}, an_code={an_code}")
    return an_code, True, int(final_x), int(final_y)

def create_style_line(style_options, video_resolution):
    """
    Create the style line for ASS subtitles.
    """
    font_family = style_options.get('font_family', 'Arial')
    available_fonts = get_available_fonts()
    if font_family not in available_fonts:
        logger.warning(f"Font '{font_family}' not found.")
        return {'error': f"Font '{font_family}' not available.", 'available_fonts': available_fonts}

    line_color = rgb_to_ass_color(style_options.get('line_color', '#FFFFFF'))
    secondary_color = line_color
    outline_color = rgb_to_ass_color(style_options.get('outline_color', '#000000'))
    box_color = rgb_to_ass_color(style_options.get('box_color', '#000000'))

    font_size = style_options.get('font_size', int(video_resolution[1] * 0.05))
    bold = '1' if style_options.get('bold', False) else '0'
    italic = '1' if style_options.get('italic', False) else '0'
    underline = '1' if style_options.get('underline', False) else '0'
    strikeout = '1' if style_options.get('strikeout', False) else '0'
    scale_x = style_options.get('scale_x', '100')
    scale_y = style_options.get('scale_y', '100')
    spacing = style_options.get('spacing', '0')
    angle = style_options.get('angle', '0')
    border_style = style_options.get('border_style', '1')
    outline_width = style_options.get('outline_width', '2')
    shadow_offset = style_options.get('shadow_offset', '0')

    margin_l = style_options.get('margin_l', '20')
    margin_r = style_options.get('margin_r', '20')
    margin_v = style_options.get('margin_v', '20')

    # Default alignment in style (we override per event)
    alignment = 5

    style_line = (
        f"Style: Default,{font_family},{font_size},{line_color},{secondary_color},"
        f"{outline_color},{box_color},{bold},{italic},{underline},{strikeout},"
        f"{scale_x},{scale_y},{spacing},{angle},{border_style},{outline_width},"
        f"{shadow_offset},{alignment},{margin_l},{margin_r},{margin_v},0"
    )
    logger.info(f"Created ASS style line: {style_line}")
    return style_line

def generate_ass_header(style_options, video_resolution):
    """
    Generate the ASS file header with the Default style.
    """
    ass_header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {video_resolution[0]}
PlayResY: {video_resolution[1]}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
"""
    style_line = create_style_line(style_options, video_resolution)
    if isinstance(style_line, dict) and 'error' in style_line:
        # Font-related error
        return style_line

    ass_header += style_line + "\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    logger.info("Generated ASS header.")
    return ass_header

### STYLE HANDLERS ###

def handle_classic(transcription_result, style_options, replace_dict, video_resolution):
    """
    Classic style handler: Centers the text based on position and alignment.
    """
    max_words_per_line = int(style_options.get('max_words_per_line', 0))
    all_caps = style_options.get('all_caps', False)
    if style_options['font_size'] is None:
        style_options['font_size'] = int(video_resolution[1] * 0.05)

    position_str = style_options.get('position', 'middle_center')
    alignment_str = style_options.get('alignment', 'center')
    x = style_options.get('x')
    y = style_options.get('y')

    an_code, use_pos, final_x, final_y = determine_alignment_code(
        position_str, alignment_str, x, y,
        video_width=video_resolution[0],
        video_height=video_resolution[1]
    )

    logger.info(f"[Classic] position={position_str}, alignment={alignment_str}, x={final_x}, y={final_y}, an_code={an_code}")

    events = []
    for segment in transcription_result['segments']:
        text = segment['text'].strip().replace('\n', ' ')
        lines = split_lines(text, max_words_per_line)
        processed_text = r'\N'.join(process_subtitle_text(line, replace_dict, all_caps, 0) for line in lines)
        start_time = format_ass_time(segment['start'])
        end_time = format_ass_time(segment['end'])
        position_tag = f"{{\\an{an_code}\\pos({final_x},{final_y})}}"
        events.append(f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{position_tag}{processed_text}")
    logger.info(f"Handled {len(events)} dialogues in classic style.")
    return "\n".join(events)

def handle_karaoke(transcription_result, style_options, replace_dict, video_resolution):
    """
    Karaoke style handler: Highlights words as they are spoken.
    """
    max_words_per_line = int(style_options.get('max_words_per_line', 0))
    all_caps = style_options.get('all_caps', False)
    if style_options['font_size'] is None:
        style_options['font_size'] = int(video_resolution[1] * 0.05)

    position_str = style_options.get('position', 'middle_center')
    alignment_str = style_options.get('alignment', 'center')
    x = style_options.get('x')
    y = style_options.get('y')

    an_code, use_pos, final_x, final_y = determine_alignment_code(
        position_str, alignment_str, x, y,
        video_width=video_resolution[0],
        video_height=video_resolution[1]
    )
    word_color = rgb_to_ass_color(style_options.get('word_color', '#FFFF00'))

    logger.info(f"[Karaoke] position={position_str}, alignment={alignment_str}, x={final_x}, y={final_y}, an_code={an_code}")

    events = []
    for segment in transcription_result['segments']:
        words = segment.get('words', [])
        if not words:
            continue

        if max_words_per_line > 0:
            lines_content = []
            current_line = []
            current_line_words = 0
            for w_info in words:
                w = process_subtitle_text(w_info.get('word', ''), replace_dict, all_caps, 0)
                duration_cs = int(round((w_info['end'] - w_info['start']) * 100))
                highlighted_word = f"{{\\k{duration_cs}}}{w} "
                current_line.append(highlighted_word)
                current_line_words += 1
                if current_line_words >= max_words_per_line:
                    lines_content.append(''.join(current_line).strip())
                    current_line = []
                    current_line_words = 0
            if current_line:
                lines_content.append(''.join(current_line).strip())
        else:
            line_content = []
            for w_info in words:
                w = process_subtitle_text(w_info.get('word', ''), replace_dict, all_caps, 0)
                duration_cs = int(round((w_info['end'] - w_info['start']) * 100))
                highlighted_word = f"{{\\k{duration_cs}}}{w} "
                line_content.append(highlighted_word)
            lines_content = [''.join(line_content).strip()]

        dialogue_text = r'\N'.join(lines_content)
        start_time = format_ass_time(words[0]['start'])
        end_time = format_ass_time(words[-1]['end'])
        position_tag = f"{{\\an{an_code}\\pos({final_x},{final_y})}}"
        events.append(f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{position_tag}{{\\c{word_color}}}{dialogue_text}")
    logger.info(f"Handled {len(events)} dialogues in karaoke style.")
    return "\n".join(events)

def handle_highlight(transcription_result, style_options, replace_dict, video_resolution):
    """
    Highlight style handler: Highlights words sequentially.
    """
    max_words_per_line = int(style_options.get('max_words_per_line', 0))
    all_caps = style_options.get('all_caps', False)
    if style_options['font_size'] is None:
        style_options['font_size'] = int(video_resolution[1] * 0.05)

    position_str = style_options.get('position', 'middle_center')
    alignment_str = style_options.get('alignment', 'center')
    x = style_options.get('x')
    y = style_options.get('y')

    an_code, use_pos, final_x, final_y = determine_alignment_code(
        position_str, alignment_str, x, y,
        video_width=video_resolution[0],
        video_height=video_resolution[1]
    )

    word_color = rgb_to_ass_color(style_options.get('word_color', '#FFFF00'))
    line_color = rgb_to_ass_color(style_options.get('line_color', '#FFFFFF'))
    events = []

    logger.info(f"[Highlight] position={position_str}, alignment={alignment_str}, x={final_x}, y={final_y}, an_code={an_code}")

    for segment in transcription_result['segments']:
        words = segment.get('words', [])
        if not words:
            continue
        processed_words = []
        for w_info in words:
            w = process_subtitle_text(w_info.get('word', ''), replace_dict, all_caps, 0)
            if w:
                processed_words.append((w, w_info['start'], w_info['end']))

        if not processed_words:
            continue

        if max_words_per_line > 0:
            line_sets = [processed_words[i:i+max_words_per_line] for i in range(0, len(processed_words), max_words_per_line)]
        else:
            line_sets = [processed_words]

        for line_set in line_sets:
            for idx, (word, w_start, w_end) in enumerate(line_set):
                line_words = []
                for w_idx, (w_text, _, _) in enumerate(line_set):
                    if w_idx == idx:
                        line_words.append(f"{{\\c{word_color}}}{w_text}{{\\c{line_color}}}")
                    else:
                        line_words.append(w_text)
                full_text = smart_join_words(line_words)
                start_time = format_ass_time(w_start)
                end_time = format_ass_time(w_end)
                position_tag = f"{{\\an{an_code}\\pos({final_x},{final_y})}}"
                events.append(f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{position_tag}{{\\c{line_color}}}{full_text}")
    logger.info(f"Handled {len(events)} dialogues in highlight style.")
    return "\n".join(events)

def handle_underline(transcription_result, style_options, replace_dict, video_resolution):
    """
    Underline style handler: Underlines the current word.
    """
    max_words_per_line = int(style_options.get('max_words_per_line', 0))
    all_caps = style_options.get('all_caps', False)
    if style_options['font_size'] is None:
        style_options['font_size'] = int(video_resolution[1] * 0.05)

    position_str = style_options.get('position', 'middle_center')
    alignment_str = style_options.get('alignment', 'center')
    x = style_options.get('x')
    y = style_options.get('y')

    an_code, use_pos, final_x, final_y = determine_alignment_code(
        position_str, alignment_str, x, y,
        video_width=video_resolution[0],
        video_height=video_resolution[1]
    )
    line_color = rgb_to_ass_color(style_options.get('line_color', '#FFFFFF'))
    events = []

    logger.info(f"[Underline] position={position_str}, alignment={alignment_str}, x={final_x}, y={final_y}, an_code={an_code}")

    for segment in transcription_result['segments']:
        words = segment.get('words', [])
        if not words:
            continue
        processed_words = []
        for w_info in words:
            w = process_subtitle_text(w_info.get('word', ''), replace_dict, all_caps, 0)
            if w:
                processed_words.append((w, w_info['start'], w_info['end']))

        if not processed_words:
            continue

        if max_words_per_line > 0:
            line_sets = [processed_words[i:i+max_words_per_line] for i in range(0, len(processed_words), max_words_per_line)]
        else:
            line_sets = [processed_words]

        for line_set in line_sets:
            for idx, (word, w_start, w_end) in enumerate(line_set):
                line_words = []
                for w_idx, (w_text, _, _) in enumerate(line_set):
                    if w_idx == idx:
                        line_words.append(f"{{\\u1}}{w_text}{{\\u0}}")
                    else:
                        line_words.append(w_text)
                full_text = smart_join_words(line_words)
                start_time = format_ass_time(w_start)
                end_time = format_ass_time(w_end)
                position_tag = f"{{\\an{an_code}\\pos({final_x},{final_y})}}"
                events.append(f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{position_tag}{{\\c{line_color}}}{full_text}")
    logger.info(f"Handled {len(events)} dialogues in underline style.")
    return "\n".join(events)

def handle_word_by_word(transcription_result, style_options, replace_dict, video_resolution):
    """
    Word-by-Word style handler: Displays each word individually.
    """
    max_words_per_line = int(style_options.get('max_words_per_line', 0))
    all_caps = style_options.get('all_caps', False)
    if style_options['font_size'] is None:
        style_options['font_size'] = int(video_resolution[1] * 0.05)

    position_str = style_options.get('position', 'middle_center')
    alignment_str = style_options.get('alignment', 'center')
    x = style_options.get('x')
    y = style_options.get('y')

    an_code, use_pos, final_x, final_y = determine_alignment_code(
        position_str, alignment_str, x, y,
        video_width=video_resolution[0],
        video_height=video_resolution[1]
    )
    word_color = rgb_to_ass_color(style_options.get('word_color', '#FFFF00'))
    events = []

    logger.info(f"[Word-by-Word] position={position_str}, alignment={alignment_str}, x={final_x}, y={final_y}, an_code={an_code}")

    for segment in transcription_result['segments']:
        words = segment.get('words', [])
        if not words:
            continue

        if max_words_per_line > 0:
            grouped_words = [words[i:i+max_words_per_line] for i in range(0, len(words), max_words_per_line)]
        else:
            grouped_words = [words]

        for word_group in grouped_words:
            for w_info in word_group:
                w = process_subtitle_text(w_info.get('word', ''), replace_dict, all_caps, 0)
                if not w:
                    continue
                start_time = format_ass_time(w_info['start'])
                end_time = format_ass_time(w_info['end'])
                position_tag = f"{{\\an{an_code}\\pos({final_x},{final_y})}}"
                events.append(f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{position_tag}{{\\c{word_color}}}{w}")
    logger.info(f"Handled {len(events)} dialogues in word-by-word style.")
    return "\n".join(events)

STYLE_HANDLERS = {
    'classic': handle_classic,
    'karaoke': handle_karaoke,
    'highlight': handle_highlight,
    'underline': handle_underline,
    'word_by_word': handle_word_by_word
}

def srt_to_ass(transcription_result, style_type, settings, replace_dict, video_resolution):
    """
    Convert transcription result to ASS based on the specified style.
    """
    default_style_settings = {
        'line_color': '#FFFFFF',
        'word_color': '#FFFF00',
        'box_color': '#000000',
        'outline_color': '#000000',
        'all_caps': False,
        'max_words_per_line': 0,
        'font_size': None,
        'font_family': 'Arial',
        'bold': False,
        'italic': False,
        'underline': False,
        'strikeout': False,
        'outline_width': 2,
        'shadow_offset': 0,
        'border_style': 1,
        'x': None,
        'y': None,
        'position': 'middle_center',
        'alignment': 'center'  # default alignment
    }
    style_options = {**default_style_settings, **settings}

    if style_options['font_size'] is None:
        style_options['font_size'] = int(video_resolution[1] * 0.05)

    ass_header = generate_ass_header(style_options, video_resolution)
    if isinstance(ass_header, dict) and 'error' in ass_header:
        # Font-related error
        return ass_header

    handler = STYLE_HANDLERS.get(style_type.lower())
    if not handler:
        logger.warning(f"Unknown style '{style_type}', defaulting to 'classic'.")
        handler = handle_classic

    dialogue_lines = handler(transcription_result, style_options, replace_dict, video_resolution)
    logger.info("Converted transcription result to ASS format.")
    return ass_header + dialogue_lines + "\n"

def process_subtitle_events(transcription_result, style_type, settings, replace_dict, video_resolution):
    """
    Process transcription results into ASS subtitle format.
    """
    return srt_to_ass(transcription_result, style_type, settings, replace_dict, video_resolution)

def process_captioning_v1(video_url, captions, settings, replace, job_id, language='auto'):
    """
    Captioning process with transcription fallback and multiple styles.
    Integrates with the updated logic for positioning and alignment.
    """
    try:
        if not isinstance(settings, dict):
            logger.error(f"Job {job_id}: 'settings' should be a dictionary.")
            return {"error": "'settings' should be a dictionary."}

        # Normalize keys by replacing hyphens with underscores
        style_options = {k.replace('-', '_'): v for k, v in settings.items()}

        if not isinstance(replace, list):
            logger.error(f"Job {job_id}: 'replace' should be a list of objects with 'find' and 'replace' keys.")
            return {"error": "'replace' should be a list of objects with 'find' and 'replace' keys."}

        # Convert 'replace' list to dictionary
        replace_dict = {}
        for item in replace:
            if 'find' in item and 'replace' in item:
                replace_dict[item['find']] = item['replace']
            else:
                logger.warning(f"Job {job_id}: Invalid replace item {item}. Skipping.")

        # Handle deprecated 'highlight_color' by merging it into 'word_color'
        if 'highlight_color' in style_options:
            logger.warning(f"Job {job_id}: 'highlight_color' is deprecated; merging into 'word_color'.")
            style_options['word_color'] = style_options.pop('highlight_color')

        # Check font availability
        font_family = style_options.get('font_family', 'Arial')
        available_fonts = get_available_fonts()
        if font_family not in available_fonts:
            logger.warning(f"Job {job_id}: Font '{font_family}' not found.")
            # Return font error with available_fonts
            return {"error": f"Font '{font_family}' not available.", "available_fonts": available_fonts}

        logger.info(f"Job {job_id}: Font '{font_family}' is available.")

        # Determine if captions is a URL or raw content
        if captions and is_url(captions):
            logger.info(f"Job {job_id}: Captions provided as URL. Downloading captions.")
            try:
                captions_content = download_captions(captions)
            except Exception as e:
                logger.error(f"Job {job_id}: Failed to download captions: {str(e)}")
                return {"error": f"Failed to download captions: {str(e)}"}
        elif captions:
            logger.info(f"Job {job_id}: Captions provided as raw content.")
            captions_content = captions
        else:
            captions_content = None

        # Download the video
        try:
            video_path = download_file(video_url, LOCAL_STORAGE_PATH)
            logger.info(f"Job {job_id}: Video downloaded to {video_path}")
        except Exception as e:
            logger.error(f"Job {job_id}: Video download error: {str(e)}")
            # For non-font errors, do NOT include available_fonts
            return {"error": str(e)}

        # Get video resolution
        video_resolution = get_video_resolution(video_path)
        logger.info(f"Job {job_id}: Video resolution detected = {video_resolution[0]}x{video_resolution[1]}")

        # Determine style type
        style_type = style_options.get('style', 'classic').lower()
        logger.info(f"Job {job_id}: Using style '{style_type}' for captioning.")

        # Determine subtitle content
        if captions_content:
            # Check if it's ASS by looking for '[Script Info]'
            if '[Script Info]' in captions_content:
                # It's ASS directly
                subtitle_content = captions_content
                subtitle_type = 'ass'
                logger.info(f"Job {job_id}: Detected ASS formatted captions.")
            else:
                # Treat as SRT
                logger.info(f"Job {job_id}: Detected SRT formatted captions.")
                # Convert SRT to transcription result with word-level timestamps
                transcription_result = srt_to_transcription_result(captions_content)
                # Generate ASS based on chosen style
                subtitle_content = process_subtitle_events(transcription_result, style_type, style_options, replace_dict, video_resolution)
                subtitle_type = 'ass'
                logger.info(f"Job {job_id}: Applied '{style_type}' style to SRT captions.")
        else:
            # No captions provided, generate transcription
            logger.info(f"Job {job_id}: No captions provided, generating transcription.")
            transcription_result = generate_transcription(video_path, language=language)
            # Generate ASS based on chosen style
            subtitle_content = process_subtitle_events(transcription_result, style_type, style_options, replace_dict, video_resolution)
            subtitle_type = 'ass'

        # Check for subtitle processing errors
        if isinstance(subtitle_content, dict) and 'error' in subtitle_content:
            logger.error(f"Job {job_id}: {subtitle_content['error']}")
            # Only include 'available_fonts' if it's a font-related error
            if 'available_fonts' in subtitle_content:
                return {"error": subtitle_content['error'], "available_fonts": subtitle_content.get('available_fonts', [])}
            else:
                return {"error": subtitle_content['error']}

        # Save the subtitle content
        subtitle_filename = f"{job_id}.{subtitle_type}"
        subtitle_path = os.path.join(LOCAL_STORAGE_PATH, subtitle_filename)
        try:
            with open(subtitle_path, 'w', encoding='utf-8') as f:
                f.write(subtitle_content)
            logger.info(f"Job {job_id}: Subtitle file saved to {subtitle_path}")
        except Exception as e:
            logger.error(f"Job {job_id}: Failed to save subtitle file: {str(e)}")
            return {"error": f"Failed to save subtitle file: {str(e)}"}

        # Prepare output filename and path
        output_filename = f"{job_id}_captioned.mp4"
        output_path = os.path.join(LOCAL_STORAGE_PATH, output_filename)

        # Process video with subtitles using FFmpeg
        try:
            # 根據字幕類型選擇適當的濾鏡
            if subtitle_type == 'ass':
                # 使用 ass 濾鏡處理 ASS 格式字幕，以支持高級樣式如高亮效果
                ffmpeg.input(video_path).output(
                    output_path,
                    vf=f"ass='{subtitle_path}'",
                    acodec='copy'
                ).run(overwrite_output=True)
                logger.info(f"Job {job_id}: Using ASS filter for advanced subtitle styles")
            else:
                # 對於其他格式使用通用的 subtitles 濾鏡
                ffmpeg.input(video_path).output(
                    output_path,
                    vf=f"subtitles='{subtitle_path}'",
                    acodec='copy'
                ).run(overwrite_output=True)
            logger.info(f"Job {job_id}: FFmpeg processing completed. Output saved to {output_path}")
        except ffmpeg.Error as e:
            stderr_output = e.stderr.decode('utf8') if e.stderr else 'Unknown error'
            logger.error(f"Job {job_id}: FFmpeg error: {stderr_output}")
            return {"error": f"FFmpeg error: {stderr_output}"}

        return output_path

    except Exception as e:
        logger.error(f"Job {job_id}: Error in process_captioning_v1: {str(e)}", exc_info=True)
        return {"error": str(e)}
