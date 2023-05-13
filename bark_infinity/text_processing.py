from typing import Dict, Optional, Union

from .config import logger, console
from typing import List
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

from typing import List
import re
import random 
from typing import Dict, Optional, Union
import logging
logger = logging.getLogger(__name__)

import re


#Let's keep comptability for now in case people are used to this
# Chunked generation originally from https://github.com/serp-ai/bark-with-voice-clone
def split_general_purpose(text, split_character_goal_length=150, split_character_max_length=200):
    # return nltk.sent_tokenize(text)

    # from https://github.com/neonbjb/tortoise-tts
    """Split text it into chunks of a desired length trying to keep sentences intact."""
    # normalize text, remove redundant whitespace and convert non-ascii quotes to ascii
    text = re.sub(r"\n\n+", "\n", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[“”]", '"', text)

    rv = []
    in_quote = False
    current = ""
    split_pos = []
    pos = -1
    end_pos = len(text) - 1

    def seek(delta):
        nonlocal pos, in_quote, current
        is_neg = delta < 0
        for _ in range(abs(delta)):
            if is_neg:
                pos -= 1
                current = current[:-1]
            else:
                pos += 1
                current += text[pos]
            if text[pos] == '"':
                in_quote = not in_quote
        return text[pos]

    def peek(delta):
        p = pos + delta
        return text[p] if p < end_pos and p >= 0 else ""

    def commit():
        nonlocal rv, current, split_pos
        rv.append(current)
        current = ""
        split_pos = []

    while pos < end_pos:
        c = seek(1)
        # do we need to force a split?
        if len(current) >= split_character_max_length:
            if len(split_pos) > 0 and len(current) > (split_character_goal_length / 2):
                # we have at least one sentence and we are over half the desired length, seek back to the last split
                d = pos - split_pos[-1]
                seek(-d)
            else:
                # should split on semicolon too
                # no full sentences, seek back until we are not in the middle of a word and split there
                while c not in ";!?.\n " and pos > 0 and len(current) > split_character_goal_length:
                    c = seek(-1)
            commit()
        # check for sentence boundaries
        elif not in_quote and (c in ";!?\n" or (c == "." and peek(1) in "\n ")):
            # seek forward if we have consecutive boundary markers but still within the max length
            while (
                pos < len(text) - 1 and len(current) < split_character_max_length and peek(1) in "!?."
            ):
                c = seek(1)
            split_pos.append(pos)
            if len(current) >= split_character_goal_length:
                commit()
        # treat end of quote as a boundary if its followed by a space or newline
        elif in_quote and peek(1) == '"' and peek(2) in "\n ":
            seek(2)
            split_pos.append(pos)
    rv.append(current)

    # clean up, remove lines with only whitespace or punctuation
    rv = [s.strip() for s in rv]
    rv = [s for s in rv if len(s) > 0 and not re.match(r"^[\s\.,;:!?]*$", s)]

    return rv

def is_sentence_ending(s):
    return s in {"!", "?", ".", ";"}

def is_boundary_marker(s):
    return s in {"!", "?", ".", "\n"}


def split_general_purpose_hm(text, split_character_goal_length=110, split_character_max_length=160):
    def clean_text(text):
        text = re.sub(r"\n\n+", "\n", text)
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[“”]", '"', text)
        return text
    
    def _split_text(text):
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
    sentences = _split_text(cleaned_text)
    wrapped_sentences = [textwrap.fill(s, width=split_character_goal_length) for s in sentences]
    chunks = [chunk for s in wrapped_sentences for chunk in s.split('\n')]
    combined_chunks = recombine_chunks(chunks)
    
    return combined_chunks



def split_text(text: str, split_type: Optional[str] = None, split_type_quantity = 1, split_type_string: Optional[str] = None, split_type_value_type: Optional[str] = None) -> List[str]:
    if text == '':
        return [text]

    # the old syntax still works if you don't use this parameter, ie
    # split_type line, split_type_value 4, splits into groups of 4 lines
    if split_type_value_type == '':
        split_type_value_type = split_type

    """
    if split_type == 'phrase':
        # print(f"Loading spacy to split by phrase.")
        nlp = spacy.load('en_core_web_sm')

        chunks = split_by_phrase(text, nlp)
        # print(chunks)
        return chunks
    """
    if split_type == 'string' or split_type == 'regex':

        if split_type_string is None:
            logger.warning(
                f"Splitting by {split_type} requires a string to split by. Returning original text.")
            return [text]

    split_type_to_function = {
        'word': split_by_words,
        'line': split_by_lines,
        'sentence': split_by_sentence,
        'string': split_by_string,
        'char' : split_by_char,
        #'random': split_by_random,
        # 'rhyme': split_by_rhymes,
        # 'pos': split_by_part_of_speech,
        'regex': split_by_regex,
    }



    if split_type in split_type_to_function:
        # split into groups of 1 by the desired type
        # this is so terrible even I'm embarassed, destroy all this code later, but I guess it does something useful atm
        segmented_text = split_type_to_function[split_type](text, split_type = split_type, split_type_quantity=1, split_type_string=split_type_string, split_type_value_type=split_type_value_type)
        final_segmented_text = []
        current_segment = ''
        split_type_quantity_found = 0

        if split_type_value_type is None:
            split_type_value_type = split_type
            
        for seg in segmented_text: # for each line, for example, we can now split by 'words' or whatever, as a counter for when to break the group
            current_segment += seg

            #print(split_type_to_function[split_type](current_segment, split_type=split_type_value_type, split_type_quantity=1, split_type_string=split_type_string))
            split_type_quantity_found = len(split_type_to_function[split_type_value_type](current_segment, split_type=split_type_value_type, split_type_quantity=1, split_type_string=split_type_string))
            #print(f"I see {split_type_quantity_found} {split_type_value_type} in {current_segment}")
            if split_type_quantity_found >= split_type_quantity:
                final_segmented_text.append(current_segment)
                split_type_quantity_found = 0
                current_segment = ''
            
        return final_segmented_text

    logger.warning(
        f"Splitting by {split_type} not a supported option. Returning original text.")
    return [text]

def split_by_string(text: str, split_type: Optional[str] = None, split_type_quantity: Optional[int] = 1, split_type_string: Optional[str] = None, split_type_value_type: Optional[str] = None) -> List[str]:
    if split_type_string is not None:
        split_pattern = f"({split_type_string})"
        split_list = re.split(split_pattern, text)
        result = [split_list[0]]
        for i in range(1, len(split_list), 2):
            result.append(split_list[i] + split_list[i+1])
        return result
    else:
        return text.split()

def split_by_regex(text: str, split_type: Optional[str] = None, split_type_quantity: Optional[int] = 1, split_type_string: Optional[str] = None, split_type_value_type: Optional[str] = None) -> List[str]:
    chunks = []
    start = 0
    if split_type_string is not None:
        for match in re.finditer(split_type_string, text):
            end = match.start()
            chunks.append(text[start:end].strip())
            start = end

        chunks.append(text[start:].strip())
        return chunks
    else:
        return text.split()

def split_by_char(text: str, split_type: Optional[str] = None, split_type_quantity = 1, split_type_string: Optional[str] = None, split_type_value_type: Optional[str] = None) -> List[str]:
    return list(text)

def split_by_words(text: str, split_type: Optional[str] = None, split_type_quantity = 1, split_type_string: Optional[str] = None, split_type_value_type: Optional[str] = None) -> List[str]:
    
    return [word + ' ' for word in text.split() if text.strip()]
    #return [' '.join(words[i:i + split_type_quantity]) for i in range(0, len(words), split_type_quantity)]


def split_by_lines(text: str, split_type: Optional[str] = None, split_type_quantity = 1, split_type_string: Optional[str] = None, split_type_value_type: Optional[str] = None) -> List[str]:
    lines = [line + '\n' for line in text.split('\n') if line.strip()]
    return lines
    #return ['\n'.join(lines[i:i + split_type_quantity]) for i in range(0, len(lines), split_type_quantity)]

def split_by_sentence(text: str, split_type: Optional[str] = None, split_type_quantity = 1, split_type_string: Optional[str] = None, split_type_value_type: Optional[str] = None) -> List[str]:
    import nltk  
    text = text.replace("\n", " ").strip()
    sentences = nltk.sent_tokenize(text)
    return [sentence + ' ' for sentence in sentences]
    #return [' '.join(sentences[i:i + split_type_quantity]) for i in range(0, len(sentences), split_type_quantity)]


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


# Good for just exploring random voices
"""
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
"""
# too many libraries, removing
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



