from bark_infinity import generation
from bark_infinity import api
from bark_infinity import SAMPLE_RATE
from bark_infinity.generation import SAMPLE_RATE, load_codec_model

from encodec.utils import convert_audio
import torchaudio
import torch
import os
import gradio
import numpy as np
import shutil

import math
import datetime
from pathlib import Path
import re
import gradio
import numpy as np


from pydub import AudioSegment



from typing import List

from math import ceil

from encodec.utils import convert_audio


from bark_infinity.hubert.customtokenizer import CustomTokenizer
from bark_infinity.hubert.customtokenizer import CustomTokenizer
from bark_infinity.hubert.hubert_manager import HuBERTManager
from bark_infinity.hubert.pre_kmeans_hubert import CustomHubert
from bark_infinity.hubert.pre_kmeans_hubert import CustomHubert
from bark_infinity.hubert.hubert_manager import HuBERTManager
import torchaudio

def sanitize_filename(filename):
    # replace invalid characters with underscores
    return re.sub(r'[^a-zA-Z0-9_]', '_', filename)



from bark_infinity.generation import (
    generate_text_semantic,
    preload_models,
    COARSE_RATE_HZ,
    SEMANTIC_RATE_HZ
)



#semantic_to_coarse_ratio = 75 / 49.9
semantic_to_coarse_ratio = COARSE_RATE_HZ / SEMANTIC_RATE_HZ



CONTEXT_WINDOW_SIZE = 1024

SEMANTIC_RATE_HZ = 49.9
SEMANTIC_VOCAB_SIZE = 10_000

CODEBOOK_SIZE = 1024
N_COARSE_CODEBOOKS = 2
N_FINE_CODEBOOKS = 8
COARSE_RATE_HZ = 75

SAMPLE_RATE = 24_000



def validate_prompt_ratio(history_prompt):
    semantic_to_coarse_ratio = COARSE_RATE_HZ / SEMANTIC_RATE_HZ

    semantic_prompt = history_prompt["semantic_prompt"]
    coarse_prompt = history_prompt["coarse_prompt"]
    fine_prompt = history_prompt["fine_prompt"]

    current_semantic_len = len(semantic_prompt)
    current_coarse_len = coarse_prompt.shape[1]
    current_fine_len = fine_prompt.shape[1]

    expected_coarse_len = int(current_semantic_len * semantic_to_coarse_ratio)
    expected_fine_len = expected_coarse_len

    if current_coarse_len != expected_coarse_len:
        print(f"Coarse length mismatch! Expected {expected_coarse_len}, got {current_coarse_len}.")
        return False

    if current_fine_len != expected_fine_len:
        print(f"Fine length mismatch! Expected {expected_fine_len}, got {current_fine_len}.")
        return False

    return True


def clone_voice(audio_filepath, dest_filename, num_clones, progress=gradio.Progress(track_tqdm=True), max_retries=1):
    old = generation.OFFLOAD_CPU
    generation.OFFLOAD_CPU = False
    progress(0, desc="HuBERT Quantizer, Quantizing.")

    # Sanitize the filename and make output directory
    dest_filename = sanitize_filename(dest_filename)
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    dir_path = Path("cloned_voices") / f"{dest_filename}_{timestamp}"
    dir_path.mkdir(parents=True, exist_ok=True)

    # Create an output path for the final output
    output_path = dir_path / f"{dest_filename}.npz"
    print(f"Cloning voice from {audio_filepath} to {dest_filename}")



	

    attempts = 0
    attempts = 0
    while attempts < max_retries:
        attempts += 1

        # Step 1: Converting WAV to Semantics
        progress(1, desc="Step 1 of 3: Converting WAV to Semantics")
        semantic_prompt_tensor = wav_to_semantics(audio_filepath)
        #semantic_prompt = semantic_prompt_tensor.numpy()
        semantic_prompt = semantic_prompt_tensor

        # Step 2: Generating Fine from WAV
        progress(2, desc="Step 2 of 3: Generating Fine from WAV")
        try:
            fine_prompt = generate_fine_from_wav(audio_filepath)
        except Exception as e:
            print(f"Failed at step 2 with error: {e}")
            continue

        # Step 3: Generating Coarse History
        progress(3, desc="Step 3 of 3: Generating Coarse History")
        coarse_prompt = generate_course_history(fine_prompt)

        # Building the history prompt
        history_prompt = {"semantic_prompt": semantic_prompt, "coarse_prompt": coarse_prompt, "fine_prompt": fine_prompt}

        # Adjust the segment size to the clone size, if semantic_prompt_len allows it.
        segment_size = len(semantic_prompt) // num_clones
        for i in range(num_clones):
            start = i * segment_size
            end = (i + 1) * segment_size if (i + 1) * segment_size <= len(semantic_prompt) else len(semantic_prompt)

            # Resize the history prompt to ensure the proper ratio
            new_semantic_len = end - start
            sliced_history_prompt = resize_history_prompt(history_prompt, tokens=new_semantic_len, from_front=True)

            if len(sliced_history_prompt["semantic_prompt"]) > 341:
                sliced_history_prompt = resize_history_prompt(sliced_history_prompt, tokens=341, from_front=False)

            # Shift semantic_prompt for the next clone
            semantic_prompt = semantic_prompt[start:end]


            #  maybe off here
            # If the resized prompt is valid, save it. Otherwise, print an error message.
            if validate_prompt_ratio(sliced_history_prompt):
                segment_output_path = output_path.with_stem(output_path.stem + f"_segment_{i}")
                api.write_seg_npz(str(segment_output_path), sliced_history_prompt) if len(sliced_history_prompt["semantic_prompt"]) > 50 else None


                if len(sliced_history_prompt["semantic_prompt"]) == 341:
                    smaller_slice = resize_history_prompt(sliced_history_prompt, tokens=192, from_front=False)
                    segment_output_path_small = output_path.with_stem(output_path.stem + f"_segment_{i}_s")
                    api.write_seg_npz(str(segment_output_path_small), smaller_slice) if len(smaller_slice["semantic_prompt"]) > 30 else None    
            else:
                current_size = len(sliced_history_prompt["semantic_prompt"])
                smaller_size = 1
                if current_size > 341 + 10:
                    smaller_size = 341
                else:
                    smaller_size = current_size - 10

                sliced_history_prompt = resize_history_prompt(sliced_history_prompt, tokens=smaller_size, from_front=False)



                if validate_prompt_ratio(sliced_history_prompt):


                    segment_output_path = output_path.with_stem(output_path.stem + f"_segment_{i}")
                    api.write_seg_npz(str(segment_output_path), sliced_history_prompt) if len(sliced_history_prompt["semantic_prompt"]) > 50 else None


                    print(f"Segment {i} not quite right but snipped")









    
        # Include an outputted history prompt that has the most possible tokens from the front, up to 341
        if len(history_prompt) > 341:
            front_history_prompt = resize_history_prompt(history_prompt, tokens=341, from_front=True)
            front_output_path = output_path.with_stem(output_path.stem + "_FIRST")
            api.write_seg_npz(str(front_output_path), front_history_prompt)

    original_audio_filepath_ext = Path(audio_filepath).suffix
    copy_of_original_target_audio_file = dir_path / f"{dest_filename}_TARGET_audio.wav"
    copy_of_original_target_audio_file = api.generate_unique_filepath(str(copy_of_original_target_audio_file))
    print(f"Copying {audio_filepath} to {copy_of_original_target_audio_file}")
    shutil.copyfile(audio_filepath, str(copy_of_original_target_audio_file))

    progress(4, desc="Finished Clone")
    print(f"Finished cloning voice from {audio_filepath} to {dest_filename}")





    progress(4, desc="Finished Clone")

    print(f"Finished cloning voice from {audio_filepath} to {dest_filename}")
    print(f"Voice clones written to: {dir_path}")

    # restore previous CPU offload state

    generation.OFFLOAD_CPU = old
    return f"Voice cloning process completed. You'll find your clones at: {dir_path}"

HuBERTManager.make_sure_hubert_installed()
HuBERTManager.make_sure_tokenizer_installed()


# checkpoint_path should work fine with data/models/hubert/hubert.pt for the default config
# https://github.com/gitmylo/bark-voice-cloning-HuBERT-quantizer  copy pasted atm 
huberts = {}
def load_hubert():
    HuBERTManager.make_sure_hubert_installed()
    HuBERTManager.make_sure_tokenizer_installed()
    if 'hubert' not in huberts:
        hubert_path = './bark_infinity/hubert/hubert.pt'
        print('Loading HuBERT')
        huberts['hubert'] = CustomHubert(hubert_path)
    if 'tokenizer' not in huberts:
        tokenizer_path  = './bark_infinity/hubert/tokenizer.pth'
        print('Loading Custom Tokenizer')
        tokenizer = CustomTokenizer()
        tokenizer.load_state_dict(torch.load(tokenizer_path))  # Load the model
        huberts['tokenizer'] = tokenizer



def generate_course_history(fine_history):
    return fine_history[:2, :]


def generate_fine_from_wav(file):
    model = load_codec_model(use_gpu=True)  # Don't worry about reimporting, it stores the loaded model in a dict
    wav, sr = torchaudio.load(file)
    wav = convert_audio(wav, sr, SAMPLE_RATE, model.channels)
    wav = wav.unsqueeze(0).to('cuda')
    with torch.no_grad():
        encoded_frames = model.encode(wav)
    codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).squeeze()

    codes = codes.cpu().numpy()

    return codes


def wav_to_semantics(file) -> torch.Tensor:
    # Vocab size is 10,000.

    load_hubert()

    wav, sr = torchaudio.load(file)
    # sr, wav = wavfile.read(file)
    # wav = torch.tensor(wav, dtype=torch.float32)

    if wav.shape[0] == 2:  # Stereo to mono if needed
        wav = wav.mean(0, keepdim=True)

    # Extract semantics in HuBERT style
    print('Extracting semantics')
    semantics = huberts['hubert'].forward(wav, input_sample_hz=sr)
    print('Tokenizing semantics')
    tokens = huberts['tokenizer'].get_token(semantics)
    
    return tokens

# Load the HuBERT model,
# checkpoint_path should work fine with data/models/hubert/hubert.pt for the default config
#hubert_model = CustomHubert(checkpoint_path='path/to/checkpoint')



import copy
from collections import Counter
import numpy as np

from contextlib import contextmanager

def load_npz(filename):
    npz_data = np.load(filename)

    data_dict = {
        "semantic_prompt": npz_data["semantic_prompt"],
        "coarse_prompt": npz_data["coarse_prompt"],
        "fine_prompt": npz_data["fine_prompt"],
    }

    npz_data.close() 

    return data_dict


def resize_history_prompt(history_prompt, tokens=128, from_front=False):
    semantic_to_coarse_ratio = COARSE_RATE_HZ / SEMANTIC_RATE_HZ

    semantic_prompt = history_prompt["semantic_prompt"]
    coarse_prompt = history_prompt["coarse_prompt"]
    fine_prompt = history_prompt["fine_prompt"]

    new_semantic_len = min(tokens, len(semantic_prompt))
    new_coarse_len = min(int(new_semantic_len * semantic_to_coarse_ratio), coarse_prompt.shape[1])
    
    new_fine_len = new_coarse_len

    if from_front:
        new_semantic_prompt = semantic_prompt[:new_semantic_len]
        new_coarse_prompt = coarse_prompt[:, :new_coarse_len]
        new_fine_prompt = fine_prompt[:, :new_fine_len]
    else:
        new_semantic_prompt = semantic_prompt[-new_semantic_len:]
        new_coarse_prompt = coarse_prompt[:, -new_coarse_len:]
        new_fine_prompt = fine_prompt[:, -new_fine_len:]

    return {
        "semantic_prompt": new_semantic_prompt,
        "coarse_prompt": new_coarse_prompt,
        "fine_prompt": new_fine_prompt,
    }


def show_history_prompt_size(history_prompt, token_samples=3, semantic_back_n=128, text="history_prompt"):

    semantic_prompt = history_prompt["semantic_prompt"]
    coarse_prompt = history_prompt["coarse_prompt"]
    fine_prompt = history_prompt["fine_prompt"]

    # compute the ratio for coarse and fine back_n
    ratio = 75 / 49.9
    coarse_and_fine_back_n = int(semantic_back_n * ratio)

    def show_array_front_back(arr, n, back_n):
        if n > 0:
            front = arr[:n].tolist()
            back = arr[-n:].tolist()

            mid = []
            if len(arr) > back_n + token_samples:
                mid = arr[-back_n-token_samples:-back_n+token_samples].tolist()

            if mid:
                return f"{front} ... <{back_n} from end> {mid} ... {back}"
            else:
                return f"{front} ... {back}"
        else:
            return ""

    def most_common_tokens(arr, n=3):
        flattened = arr.flatten()
        counter = Counter(flattened)
        return counter.most_common(n)

    print(f"\n{text}")
    print(f"  {text} semantic_prompt: {semantic_prompt.shape}")
    print(f"    Tokens: {show_array_front_back(semantic_prompt, token_samples, semantic_back_n)}")
    print(f"    Most common tokens: {most_common_tokens(semantic_prompt)}")
    
    print(f"  {text} coarse_prompt: {coarse_prompt.shape}")
    for i, row in enumerate(coarse_prompt):
        print(f"    Row {i} Tokens: {show_array_front_back(row, token_samples, coarse_and_fine_back_n)}")
        print(f"    Most common tokens in row {i}: {most_common_tokens(row)}")
    
    print(f"  {text} fine_prompt: {fine_prompt.shape}")
    #for i, row in enumerate(fine_prompt):
        #print(f"    Row {i} Tokens: {show_array_front_back(row, token_samples, coarse_and_fine_back_n)}")
        #print(f"    Most common tokens in row {i}: {most_common_tokens(row)}")




def split_array_equally(array, num_parts):
    split_indices = np.linspace(0, len(array), num_parts + 1, dtype=int)
    return [array[split_indices[i]: split_indices[i + 1]].astype(np.int32) for i in range(num_parts)]




@contextmanager
def measure_time(text=None, index=None):
    start_time = time.time()
    yield
    elapsed_time = time.time() - start_time
    if index is not None and text is not None:
        text = f"{text} {index}"
    elif text is None:
        text = "Operation"
    
    time_finished = f"{text} Finished at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}"
    print(f"  -->{time_finished} in {elapsed_time} seconds")



def compare_history_prompts(hp1, hp2, text="history_prompt"):
    print(f"\nComparing {text}")
    for key in hp1.keys():
        if hp1[key].shape != hp2[key].shape:
            print(f"  {key} arrays have different shapes: {hp1[key].shape} vs {hp2[key].shape}.")
            min_size = min(hp1[key].shape[0], hp2[key].shape[0])

            if hp1[key].ndim == 1:
                hp1_part = hp1[key][-min_size:]
                hp2_part = hp2[key][-min_size:]
            else:
                min_size = min(hp1[key].shape[1], hp2[key].shape[1])
                hp1_part = hp1[key][:, -min_size:]
                hp2_part = hp2[key][:, -min_size:]
            
            print(f"  Comparing the last {min_size} elements of each.")
        else:
            hp1_part = hp1[key]
            hp2_part = hp2[key]

        if np.array_equal(hp1_part, hp2_part):
            print(f"    {key} arrays are exactly the same.")
        elif np.allclose(hp1_part, hp2_part):
            diff = np.linalg.norm(hp1_part - hp2_part)
            print(f"    {key} arrays are almost equal with a norm of difference: {diff}")
        else:
            diff = np.linalg.norm(hp1_part - hp2_part)
            print(f"    {key} arrays are not equal. Norm of difference: {diff}")

            


def split_by_words(text, word_group_size):
    words = text.split()
    result = []
    group = ""
    
    for i, word in enumerate(words):
        group += word + " "
        
        if (i + 1) % word_group_size == 0:
            result.append(group.strip())
            group = ""
    
    # Add the last group if it's not empty
    if group.strip():
        result.append(group.strip())
    
    return result

def concat_history_prompts(history_prompt1, history_prompt2):
    new_semantic_prompt = np.hstack([history_prompt1["semantic_prompt"], history_prompt2["semantic_prompt"]]).astype(np.int32) #not int64?
    new_coarse_prompt = np.hstack([history_prompt1["coarse_prompt"], history_prompt2["coarse_prompt"]]).astype(np.int32)
    new_fine_prompt = np.hstack([history_prompt1["fine_prompt"], history_prompt2["fine_prompt"]]).astype(np.int32)

    concatenated_history_prompt = {
        "semantic_prompt": new_semantic_prompt,
        "coarse_prompt": new_coarse_prompt,
        "fine_prompt": new_fine_prompt,
    }

    return concatenated_history_prompt


def merge_history_prompts(left_history_prompt, right_history_prompt, right_size = 128):
    right_history_prompt = resize_history_prompt(right_history_prompt, tokens=right_size, from_front=False)
    combined_history_prompts = concat_history_prompts(left_history_prompt, right_history_prompt)
    combined_history_prompts = resize_history_prompt(combined_history_prompts, tokens=341, from_front=False)
    return combined_history_prompts
