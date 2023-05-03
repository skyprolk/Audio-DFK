from typing import Dict, Optional, Union

from .config import logger, console

import os
import re
import datetime
import random
from typing import List

import re
import textwrap


from rich.pretty import pprint
from rich.table import Table


from collections import defaultdict

def is_sentence_ending(s):
    return s in {"!", "?", "."}

def is_boundary_marker(s):
    return s in {"!", "?", ".", "\n"}

# I started with this function from tortoise, I integrated Spacy and a bunch of other stuff, but it was a headache and it was especially bad with other languages, so I went back to this simple version and cleaned it up a bit
def split_general_purpose(text, split_character_goal_length=110, split_character_max_length=160):
    def clean_text(text):
        text = re.sub(r"\n\n+", "\n", text)
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[“”]", '"', text)
        return text
    
    def split_text(text):
        sentences = []
        sentence = ""
        in_quote = False
        for i, c in enumerate(text):
            sentence += c
            if c == '"':
                in_quote = not in_quote
            elif not in_quote and (is_sentence_ending(c) or c == "\n"):
                if i < len(text) - 1 and text[i + 1] in '!?.':
                    continue
                sentences.append(sentence.strip())
                sentence = ""
        if sentence.strip():
            sentences.append(sentence.strip())
        return sentences

    def recombine_chunks(chunks):
        combined_chunks = []
        current_chunk = ""
        for chunk in chunks:
            if len(current_chunk) + len(chunk) + 1 <= split_character_max_length:
                current_chunk += " " + chunk
            else:
                combined_chunks.append(current_chunk.strip())
                current_chunk = chunk
        if current_chunk.strip():
            combined_chunks.append(current_chunk.strip())
        return combined_chunks

    cleaned_text = clean_text(text)
    sentences = split_text(cleaned_text)
    wrapped_sentences = [textwrap.fill(s, width=split_character_goal_length) for s in sentences]
    chunks = [chunk for s in wrapped_sentences for chunk in s.split('\n')]
    combined_chunks = recombine_chunks(chunks)
    
    return combined_chunks


# This is just a big ol mess of code, remove later
def split_text(text: str = "", split_type: str = None, n='') -> List[str]:
    if n is None or split_type is None:
        return [text]

    """
    if split_type == 'phrase':
        # print(f"Loading spacy to split by phrase.")
        nlp = spacy.load('en_core_web_sm')

        chunks = split_by_phrase(text, nlp)
        # print(chunks)
        return chunks
    """
    if split_type != 'string' and split_type != 'regex' and split_type != 'pos':
        n = int(n)
    split_type_to_function = {
        'word': split_by_words,
        'line': split_by_lines,
        # 'sentence': split_by_sentences,
        'string': split_by_string,
        'random': split_by_random,
        # 'rhyme': split_by_rhymes,
        # 'pos': split_by_part_of_speech,
        'regex': split_by_regex,
    }

    if split_type in split_type_to_function:
        return split_type_to_function[split_type](text, n)

    logger.warning(
        f"Splitting by {split_type} not a supported option. Returning original text.")
    return [text]


def split_by_string(text: str, separator: str) -> List[str]:
    return text.split(separator)


def split_by_regex(text: str, pattern: str) -> List[str]:
    chunks = []
    start = 0

    for match in re.finditer(pattern, text):
        end = match.start()
        chunks.append(text[start:end].strip())
        start = end

    chunks.append(text[start:].strip())
    return chunks


def split_by_words(text: str, n: int) -> List[str]:
    words = text.split()
    return [' '.join(words[i:i + n]) for i in range(0, len(words), n)]








def split_by_lines(text: str, n: int) -> List[str]:
    lines = [line for line in text.split('\n') if line.strip()]
    return ['\n'.join(lines[i:i + n]) for i in range(0, len(lines), n)]

"""
def split_by_sentences(text: str, n: int, language="en") -> List[str]:
    seg = pysbd.Segmenter(language=language, clean=False)
    sentences = seg.segment(text)
    return [' '.join(sentences[i:i + n]) for i in range(0, len(sentences), n)]
"""

def load_text(file_path: str) -> Union[str, None]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        logger.info(f"Successfully loaded the file: {file_path}")
        return content
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
    except PermissionError:
        logger.error(f"Permission denied to read the file: {file_path}")
    except Exception as e:
        logger.error(
            f"An unexpected error occurred while reading the file: {file_path}. Error: {e}")
    return None


def split_by_random(text: str, n: int) -> List[str]:
    words = text.split()
    chunks = []
    min_len = max(1, n - 2)
    max_len = n + 2
    while words:
        chunk_len = random.randint(min_len, max_len)
        chunk = ' '.join(words[:chunk_len])
        chunks.append(chunk)
        words = words[chunk_len:]
    return chunks

# too many libraries, removing for public release
"""
def split_by_phrase(text: str, nlp, min_duration=8, max_duration=18, words_per_second=2.3) -> list:

    if text is None:
        return ''
    doc = nlp(text)
    chunks = []
    min_words = int(min_duration * words_per_second)
    max_words = int(max_duration * words_per_second)

    current_chunk = ""
    current_word_count = 0

    for sent in doc.sents:
        word_count = len(sent.text.split())
        if current_word_count + word_count < min_words:
            current_chunk += " " + sent.text.strip()
            current_word_count += word_count
        elif current_word_count + word_count <= max_words:
            current_chunk += " " + sent.text.strip()
            chunks.append(current_chunk.strip())
            current_chunk = ""
            current_word_count = 0
        else:
            # Emergency cutoff
            words = sent.text.split()
            while words:
                chunk_len = max_words - current_word_count
                chunk = ' '.join(words[:chunk_len])
                current_chunk += " " + chunk
                chunks.append(current_chunk.strip())
                current_chunk = ""
                current_word_count = 0
                words = words[chunk_len:]

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks
"""

"""
def split_by_rhymes(text: str, n: int) -> List[str]:
    words = text.split()
    chunks = []
    current_chunk = []
    rhyming_word_count = 0
    for word in words:
        current_chunk.append(word)
        if any(rhyme_word in words for rhyme_word in rhymes(word)):
            rhyming_word_count += 1
            if rhyming_word_count >= n:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                rhyming_word_count = 0
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks
"""

# 'NN' for noun. 'VB' for verb. 'JJ' for adjective. 'RB' for adverb.
# NN-VV Noun followed by a verb
# JJR, JJS
# UH = Interjection, Goodbye Goody Gosh Wow Jeepers Jee-sus Hubba Hey Kee-reist Oops amen huh howdy uh dammit whammo shucks heck anyways whodunnit honey golly man baby diddle hush sonuvabitch ...

"""
def split_by_part_of_speech(text: str, pos_pattern: str) -> List[str]:
    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)
    pos_pattern = pos_pattern.split('-')
    original_pos_pattern = pos_pattern.copy()

    chunks = []
    current_chunk = []

    for word, pos in tagged_tokens:
        current_chunk.append(word)
        if pos in pos_pattern:
            pos_index = pos_pattern.index(pos)
            if pos_index == 0:
                pos_pattern.pop(0)
            else:
                current_chunk = current_chunk[:-1]
                pos_pattern = original_pos_pattern.copy()
        if not pos_pattern:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            pos_pattern = original_pos_pattern.copy()

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks
"""



