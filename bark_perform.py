import argparse
import numpy as np

from rich import print

from bark_infinity import config

logger = config.logger

from bark_infinity import generation
from bark_infinity import api

from bark_infinity import text_processing
import time

import random

text_prompts_in_this_file = []


import torch
from torch.utils import collect_env


try:
    text_prompts_in_this_file.append(
        f"It's {text_processing.current_date_time_in_words()} And if you're hearing this, Bark is working. But you didn't provide any text"
    )
except Exception as e:
    print(f"An error occurred: {e}")

text_prompt = """
    In the beginning the Universe was created. This has made a lot of people very angry and been widely regarded as a bad move. However, Bark is working.
"""
text_prompts_in_this_file.append(text_prompt)

text_prompt = """
    A common mistake that people make when trying to design something completely foolproof is to underestimate the ingenuity of complete fools.
"""
text_prompts_in_this_file.append(text_prompt)


def get_group_args(group_name, updated_args):
    # Convert the Namespace object to a dictionary
    updated_args_dict = vars(updated_args)

    group_args = {}
    for key, value in updated_args_dict.items():
        if key in dict(config.DEFAULTS[group_name]):
            group_args[key] = value
    return group_args


def main(args):
    if args.loglevel is not None:
        logger.setLevel(args.loglevel)

    if args.OFFLOAD_CPU is not None:
        generation.OFFLOAD_CPU = args.OFFLOAD_CPU
        # print(f"OFFLOAD_CPU is set to {generation.OFFLOAD_CPU}")
    else:
        if generation.get_SUNO_USE_DIRECTML() is not True:
            generation.OFFLOAD_CPU = True  # default on just in case
    if args.USE_SMALL_MODELS is not None:
        generation.USE_SMALL_MODELS = args.USE_SMALL_MODELS
        # print(f"USE_SMALL_MODELS is set to {generation.USE_SMALL_MODELS}")
    if args.GLOBAL_ENABLE_MPS is not None:
        generation.GLOBAL_ENABLE_MPS = args.GLOBAL_ENABLE_MPS
        # print(f"GLOBAL_ENABLE_MPS is set to {generation.GLOBAL_ENABLE_MPS}")

    if not args.silent:
        if args.detailed_gpu_report:
            print(api.startup_status_report(quick=False))
        elif not args.text_prompt and not args.prompt_file:  # probably a test run, default to show
            print(api.startup_status_report(quick=True))
        if args.detailed_hugging_face_cache_report:
            print(api.hugging_face_cache_report())
        if args.detailed_cuda_report:
            print(api.cuda_status_report())

    if args.list_speakers:
        api.list_speakers()
        return

    if args.render_npz_samples:
        api.render_npz_samples()
        return

    if args.text_prompt:
        text_prompts_to_process = [args.text_prompt]
    elif args.prompt_file:
        text_file = text_processing.load_text(args.prompt_file)
        if text_file is None:
            logger.error(f"Error loading file: {args.prompt_file}")
            return
        text_prompts_to_process = text_processing.split_text(
            text_processing.load_text(args.prompt_file),
            args.split_input_into_separate_prompts_by,
            args.split_input_into_separate_prompts_by_value,
        )


        print(f"\nProcessing file: {args.prompt_file}")
        print(f"  Looks like: {len(text_prompts_to_process)} prompt(s)")

    else:
        print("No --text_prompt or --prompt_file specified, using test prompt.")
        text_prompts_to_process = random.sample(text_prompts_in_this_file, 2)

    things = len(text_prompts_to_process) + args.output_iterations
    if things > 10:
        if args.dry_run is False:
            print(
                f"WARNING: You are about to process {things} prompts. Consider using '--dry-run' to test things first."
            )

    # pprint(args)
    print("Loading Bark models...")
    if not args.dry_run and generation.get_SUNO_USE_DIRECTML() is not True:
        generation.preload_models(
            args.text_use_gpu,
            args.text_use_small,
            args.coarse_use_gpu,
            args.coarse_use_small,
            args.fine_use_gpu,
            args.fine_use_small,
            args.codec_use_gpu,
            args.force_reload,
        )

    print("Done.")

    for idx, text_prompt in enumerate(text_prompts_to_process, start=1):
        if len(text_prompts_to_process) > 1:
            print(f"\nPrompt {idx}/{len(text_prompts_to_process)}:")

        # print(f"Text prompt: {text_prompt}")
        for iteration in range(1, args.output_iterations + 1):
            if args.output_iterations > 1:
                print(f"\nIteration {iteration} of {args.output_iterations}.")
                if iteration == 1:
                    print("ss", text_prompt)

            args.current_iteration = iteration
            args.text_prompt = text_prompt
            args_dict = vars(args)

            api.generate_audio_long(**args_dict)


if __name__ == "__main__":
    parser = config.create_argument_parser()

    args = parser.parse_args()

    updated_args = config.update_group_args_with_defaults(args)

    namespace_args = argparse.Namespace(**updated_args)
    main(namespace_args)
