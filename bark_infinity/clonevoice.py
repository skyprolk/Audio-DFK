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






CONTEXT_WINDOW_SIZE = 1024

SEMANTIC_RATE_HZ = 49.9
SEMANTIC_VOCAB_SIZE = 10_000

CODEBOOK_SIZE = 1024
N_COARSE_CODEBOOKS = 2
N_FINE_CODEBOOKS = 8
COARSE_RATE_HZ = 75

SAMPLE_RATE = 24_000

TEXT_ENCODING_OFFSET = 10_048
SEMANTIC_PAD_TOKEN = 10_000
TEXT_PAD_TOKEN = 129_595
SEMANTIC_INFER_TOKEN = 129_599

from bark_infinity import api
from bark_infinity import generation
from bark_infinity import text_processing
from bark_infinity import config


"""
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
"""
import os
def write_clone_npz(filepath, full_generation, regen_fine = False, **kwargs):



    filepath = api.generate_unique_filepath(filepath)
    #np.savez_compressed(filepath, semantic_prompt = full_generation["semantic_prompt"], coarse_prompt = full_generation["coarse_prompt"], fine_prompt = full_generation["fine_prompt"])
    if "semantic_prompt" in full_generation:
        np.savez(filepath, semantic_prompt = full_generation["semantic_prompt"], coarse_prompt = full_generation["coarse_prompt"], fine_prompt = full_generation["fine_prompt"])
        quick_codec_render(filepath)
    else:
        print("No semantic prompt to save")

    if regen_fine:
        # purposely not passing any speaker history here, not super helpful though fine still doesn't matter
        fine_tokens = generation.generate_fine(full_generation["coarse_prompt"])
        base = os.path.basename(filepath)
        filename, extension = os.path.splitext(base)
        suffix = "a"
        new_filename = filename + suffix
        new_filepath = os.path.join(os.path.dirname(filepath), new_filename + extension)
        new_filepath = api.generate_unique_filepath(new_filepath)
        np.savez(new_filepath, semantic_prompt = full_generation["semantic_prompt"], coarse_prompt = full_generation["coarse_prompt"], fine_prompt = fine_tokens)


def clone_voice(audio_filepath, input_audio_filename_secondary, dest_filename, semantic_step_interval = 192, progress=gradio.Progress(track_tqdm=True), max_retries=1, even_more_clones=False):
    old = generation.OFFLOAD_CPU
    generation.OFFLOAD_CPU = False
    progress(0, desc="HuBERT Quantizer, Quantizing.")

 
    dest_filename = sanitize_filename(dest_filename)
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    dir_path = Path("cloned_voices") / f"{dest_filename}_{timestamp}"
    dir_path.mkdir(parents=True, exist_ok=True)

    base_output_path = dir_path / f"{dest_filename}.npz"
    print(f"Cloning voice from {audio_filepath} to {dest_filename}")


    default_prompt_width = 512

    budget_prompt_width = 256

    attempts = 0

    orig_semantic_prompt = None
    all_completed_clones = []
    while attempts < max_retries:
        attempts += 1


        # Step 1: Converting WAV to Semantics
        progress(1, desc="Step 1 of 4: Converting WAV to Semantics")
        semantic_prompt_tensor = wav_to_semantics(audio_filepath)
        orig_semantic_prompt = semantic_prompt_tensor
        #semantic_prompt = semantic_prompt_tensor.numpy()
        semantic_prompt = semantic_prompt_tensor

        # Step 2: Generating Fine from WAV
        progress(2, desc="Step 2 of 4: Generating Fine from WAV")
        try:
            fine_prompt = generate_fine_from_wav(audio_filepath)
        except Exception as e:
            print(f"Failed at step 2 with error: {e}")
            continue

        # Step 3: Generating Coarse History
        progress(3, desc="Step 3 of 4: Generating Coarse History")
        coarse_prompt = generate_course_history(fine_prompt)
        #coarse_prompt = coarse_prompt.numpy()

        # Building the history prompt
        history_prompt = {"semantic_prompt": semantic_prompt, "coarse_prompt": coarse_prompt, "fine_prompt": fine_prompt}

        # print types of each
        #print(f"semantic_prompt type: {type(semantic_prompt)}")
        #print(f"coarse_prompt type: {type(coarse_prompt)}")
        #print(f"fine_prompt type: {type(fine_prompt)}")


        if not api.history_prompt_is_valid(history_prompt):
            print("Primary prompt potentially problematic:")
            print(api.history_prompt_detailed_report(history_prompt))
            

        attempt_string = f"_{attempts}"


        output_path = base_output_path.with_stem(base_output_path.stem + attempt_string)

        #full_output_path = output_path.with_stem(output_path.stem + "_FULLAUDIOCLIP")
        #write_clone_npz(str(full_output_path), history_prompt)


        # The back of audio is generally the best speaker by far, as the user specifically chose this audio clip and it likely has a natural ending.
        # If you had to choose one the front of the clip is bit different style and decent, though cutting randomly so 
        # it has a high chance of being terrible. 


       



        # full_output_path = output_path.with_stem(output_path.stem + "_FULLAUDIOCLIP")
        # write_clone_npz(str(full_output_path), history_prompt)


        # The back of clip generally the best speaker, as the user specifically chose this audio clip and it likely has a natural ending.
        # the front of the clip is often a quite different style and decent, though we're cutting randomly so 
        # it has a high chance of being terrible. Still let's surface it higher, if we feature any other one in particular

        clip_full_semantic_length = len(semantic_prompt)

        back_history_prompt = resize_history_prompt(history_prompt, tokens=341, from_front=False)
        back_output_path = output_path.with_stem(output_path.stem + "__MAIN")
        write_clone_npz(str(back_output_path), back_history_prompt, regen_fine=False)
        all_completed_clones.append((back_history_prompt, str(back_output_path), clip_full_semantic_length - 341))

        if clip_full_semantic_length > 341:

            front_history_prompt = resize_history_prompt(history_prompt, tokens=341, from_front=True)
            front_output_path = output_path.with_stem(output_path.stem + "_FRONT")
            write_clone_npz(str(front_output_path), front_history_prompt, regen_fine=False)
            all_completed_clones.append((front_history_prompt, str(front_output_path), 341))


        """
        clip_full_semantic_length = len(semantic_prompt)

        back_history_prompt = resize_history_prompt(history_prompt, tokens=default_prompt_width, from_front=False)
        back_output_path = output_path.with_stem(output_path.stem + "__MAIN")
        write_clone_npz(str(back_output_path), back_history_prompt, regen_fine=False)
        all_completed_clones.append((back_history_prompt, str(back_output_path), clip_full_semantic_length - default_prompt_width))

        if clip_full_semantic_length > budget_prompt_width:

            front_history_prompt = resize_history_prompt(history_prompt, tokens=budget_prompt_width, from_front=True)
            front_output_path = output_path.with_stem(output_path.stem + "_FRONT")
            write_clone_npz(str(front_output_path), front_history_prompt, regen_fine=False)
            all_completed_clones.append((front_history_prompt, str(front_output_path), budget_prompt_width))

        """
        """
        print("raw")
        print(api.history_prompt_detailed_report(history_prompt))
        print("loaded")
        print(api.history_prompt_detailed_report(str(output_path)))
        """

        clone_start = semantic_step_interval
        clone_end = clone_start + budget_prompt_width

        # print(f"full clip length: {clip_full_semantic_length}")
        segment_number = 0
        while clone_end < clip_full_semantic_length + semantic_step_interval:
            segment_number += 1
            current_slice = clone_end
            clone_end += semantic_step_interval
            sliced_history_prompt = resize_history_prompt(history_prompt, tokens=current_slice, from_front=True)
            sliced_history_prompt = resize_history_prompt(sliced_history_prompt, tokens=budget_prompt_width, from_front=False)
            if api.history_prompt_is_valid(sliced_history_prompt):
                segment_output_path = output_path.with_stem(output_path.stem + f"_s_{segment_number}")
            else:
                print(f"segment {segment_number} potentially problematic:")
                print(api.history_prompt_detailed_report(sliced_history_prompt))
                sliced_history_prompt = resize_history_prompt(sliced_history_prompt, tokens=budget_prompt_width - 1, from_front=False)
                if api.history_prompt_is_valid(sliced_history_prompt):
                    segment_output_path = output_path.with_stem(output_path.stem + f"_s_{segment_number}")
                else:
                    print(f"segment {segment_number} still potentially problematic:")
                    print(api.history_prompt_detailed_report(sliced_history_prompt))
                    continue

            write_clone_npz(str(segment_output_path), sliced_history_prompt, regen_fine=False)
            all_completed_clones.append((sliced_history_prompt, str(segment_output_path),current_slice ))


    original_audio_filepath_ext = Path(audio_filepath).suffix
    copy_of_original_target_audio_file = dir_path / f"{dest_filename}_TARGET_audio.wav"
    copy_of_original_target_audio_file = api.generate_unique_filepath(str(copy_of_original_target_audio_file))
    print(f"Copying {audio_filepath} to {copy_of_original_target_audio_file}")
    shutil.copyfile(audio_filepath, str(copy_of_original_target_audio_file))

    progress(4, desc="Quick Clones Done")
    print(f"Finished cloning voice from {audio_filepath} to {dest_filename}")

    # TODO just an experiment, doesn't seem to help though
    orig_semantic_prompt = orig_semantic_prompt.numpy()

    import random
    if even_more_clones or input_audio_filename_secondary is not None:
        progress(5, desc="Generative Clones, Long Clip")




        second_sample_prompt = None
        if input_audio_filename_secondary is not None:
            progress(5, desc="Step 5 of 5: Converting Secondary Audio sample to Semantic Prompt")
            second_sample_tensor = wav_to_semantics(input_audio_filename_secondary)
            second_sample_prompt = second_sample_tensor.numpy()
            if len(second_sample_prompt) > 850:
                second_sample_prompt = second_sample_prompt[:850] # Actuall from front, makes sense


        orig_semantic_prompt_len = len(orig_semantic_prompt)


        total_clones = len(all_completed_clones)
        clone_num = 0
        for clone, filepath, end_slice in all_completed_clones:
            clone_num += 1
            clone_history = load_npz(filepath) # lazy tensor to numpy...
            progress(5, desc=f"Generating {clone_num} of {total_clones}")
            if api.history_prompt_is_valid(clone_history):


                end_of_prompt = end_slice + budget_prompt_width
                if end_of_prompt > orig_semantic_prompt_len:
                    semantic_next_segment = orig_semantic_prompt # use beginning 
                else:
                    semantic_next_segment = orig_semantic_prompt[-(orig_semantic_prompt_len - end_slice):]


                prompts = []
                if second_sample_prompt is not None:
                    prompts.append(second_sample_prompt)    
                
                if even_more_clones:
                    prompts.append(semantic_next_segment)


      
                for semantic_next_segment in prompts:
                    
                    #print(f"Shape of semantic_next_segment: {semantic_next_segment.shape}")

                    if len(semantic_next_segment) > 800:
                        semantic_next_segment = semantic_next_segment[:800]

                    chop_sizes = [32, 64, 96, 128, 192, 256]

                    chop = random.choice(chop_sizes)

                    if chop == 0:
                        chop_his = None
                    else: 
                        chop_his = resize_history_prompt(clone_history, tokens=chop, from_front=False)
                    coarse_tokens = api.generate_coarse(
                        semantic_next_segment,
                        history_prompt=chop_his,
                        temp=0.7,
                        silent=False,
                        use_kv_caching=True,
                    )

                    fine_tokens = api.generate_fine(
                        coarse_tokens,
                        history_prompt=chop_his,
                        temp=0.5,
                    )


                    full_generation = {
                        "semantic_prompt": semantic_next_segment,
                        "coarse_prompt": coarse_tokens,
                        "fine_prompt": fine_tokens,
                    }
                    
                    if api.history_prompt_is_valid(full_generation):

                        base = os.path.basename(filepath)
                        filename, extension = os.path.splitext(base)
                        suffix = f"g2_{chop}_"
                        new_filename = filename + suffix
                        new_filepath = os.path.join(os.path.dirname(filepath), new_filename + extension)
                        new_filepath = api.generate_unique_filepath(new_filepath)
                        write_clone_npz(new_filepath, full_generation)

                        # messy, really bark infinity should sample from different spaces in huge npz files, no reason to cut like this.
                        suffix = f"g2f_{chop}_"
                        full_generation = resize_history_prompt(full_generation, tokens=budget_prompt_width, from_front=True)
                        new_filename = filename + suffix
                        new_filepath = os.path.join(os.path.dirname(filepath), new_filename + extension)
                        new_filepath = api.generate_unique_filepath(new_filepath)
                        write_clone_npz(new_filepath, full_generation)


                        tiny_history_addition = resize_history_prompt(full_generation, tokens=128, from_front=True)
                        merged = merge_history_prompts(chop_his, tiny_history_addition, right_size = 128)
                        suffix = f"g2t_{chop}_"
                        full_generation = resize_history_prompt(merged, tokens=budget_prompt_width, from_front=False)
                        new_filename = filename + suffix
                        new_filepath = os.path.join(os.path.dirname(filepath), new_filename + extension)
                        new_filepath = api.generate_unique_filepath(new_filepath)
                        write_clone_npz(new_filepath, full_generation)
                    else: 
                        print(f"Full generation for {filepath} was invalid, skipping")
                        print(api.history_prompt_detailed_report(full_generation))
            else:
                print(f"Clone {filepath} was invalid, skipping")
                print(api.history_prompt_detailed_report(clone_history))


    print(f"Voice cloning process completed. You'll find your clones at: {dir_path}")

    # restore previous CPU offload state

    generation.OFFLOAD_CPU = old
    return f"{dir_path}"




def quick_codec_render(filepath):
    reload = load_npz(filepath) # lazy 
    fine_prompt = reload['fine_prompt']
    audio_arr = generation.codec_decode(fine_prompt)


    base = os.path.basename(filepath)
    filename, extension = os.path.splitext(base)
    new_filepath = os.path.join(os.path.dirname(filepath), filename + '_raw.mp3')
    new_filepath = api.generate_unique_filepath(new_filepath)
    api.write_audiofile(new_filepath, audio_arr, output_format='mp3')

"""

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
"""

huberts = {}

bark_cloning_large_model = True # 
def load_hubert():
    hubert_path = HuBERTManager.make_sure_hubert_installed()
    model = ('quantifier_V1_hubert_base_ls960_23.pth', 'tokenizer_large.pth') if bark_cloning_large_model else ('quantifier_hubert_base_ls960_14.pth', 'tokenizer.pth')
    tokenizer_path = HuBERTManager.make_sure_tokenizer_installed(model=model[0], local_file=model[1])
    if 'hubert' not in huberts:
        print(f'Loading HuBERT models {model} from {hubert_path}')
        huberts['hubert'] = CustomHubert(hubert_path)
    if 'tokenizer' not in huberts:
        print('Loading Custom Tokenizer')
        print(f'Loading tokenizer from {tokenizer_path}')
        tokenizer = CustomTokenizer.load_from_checkpoint(tokenizer_path, map_location=torch.device('cpu'))
        huberts['tokenizer'] = tokenizer


def generate_course_history(fine_history):
    return fine_history[:2, :]


# TODO don't hardcode GPU
"""
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
"""
clone_use_gpu = False
def generate_fine_from_wav(file):
    #model = load_codec_model(use_gpu=not args.bark_use_cpu)  # Don't worry about reimporting, it stores the loaded model in a dict
    model = load_codec_model(use_gpu=False)  # Don't worry about reimporting, it stores the loaded model in a dict
    wav, sr = torchaudio.load(file)
    wav = convert_audio(wav, sr, SAMPLE_RATE, model.channels)
    wav = wav.unsqueeze(0)
    #if not (args.bark_cpu_offload or args.bark_use_cpu):
    if (False):
        wav = wav.to('cuda')
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
