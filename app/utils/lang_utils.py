# app/utils/lang_utils.py
import logging

logger = logging.getLogger(__name__)

# 한글 자모 및 완성형 유니코드 범위 (가-힣, ㄱ-ㅎ, ㅏ-ㅣ)
# 참고: https://en.wikipedia.org/wiki/Hangul_Syllables, https://en.wikipedia.org/wiki/Hangul_Jamo_(Unicode_block)
_HANGUL_START = 0xAC00
_HANGUL_END = 0xD7A3
_HANGUL_JAMO_START = 0x1100
_HANGUL_JAMO_END = 0x11FF

_HANGUL_RANGES = [
    (_HANGUL_START, _HANGUL_END),
    (_HANGUL_JAMO_START, _HANGUL_JAMO_END),
]

def contains_hangul(text: str) -> bool:
    """
    Checks if the given text contains any Hangul (Korean) characters
    within the common Syllables and Jamo Unicode ranges.
    Uses a simple O(N) character iteration.
    """
    if not text:
        return False

    for char in text:
        code = ord(char)
        for start, end in _HANGUL_RANGES:
            if start <= code <= end:
                return True
    return False

def classify_language(text: str) -> str:
    """
    Classifies the language of the text as 'ko' if it contains Hangul,
    otherwise defaults to 'en'.
    """
    if contains_hangul(text):
        return 'ko'
    else:
        return 'en' 