import datetime
import os
import random
import glob
import gradio as gr
from gradio.components import Markdown as m
import sys
from collections import defaultdict
from tqdm import tqdm
os.environ["TERM"] = "dumb"
import io
from bark_infinity import config

logger = config.logger
logger.setLevel("INFO")

from bark_infinity import generation
from bark_infinity import api


generation.OFFLOAD_CPU = True
generation.USE_SMALL_MODELS = False

base_theme = gr.themes.Base()
default_theme = gr.themes.Default()
monochrome_theme = gr.themes.Monochrome()
soft_theme = gr.themes.Soft()
glass_theme = gr.themes.Glass()


# If anyone is looking at this code, I just took gradio blocks kitchen sink demo and cut and pasted all over the place, and as usual I know realize I should have just carefully read the Gradio explanation from the beginning. 

from bark_infinity.clonevoice import clone_voice

import threading
import time

# this is the 'ripped from Stable Diffusion' section because that's where the only place I could find all the Gradio examples for stuff like this
from webui import styles
from webui import transformations
from webui.ui_components import FormRow, FormColumn, FormGroup, ToolButton, FormHTML

style_csv = "webui/styles.csv"
user_style_csv = "webui/user_styles.csv"

transformation_csv = "webui/transformations.csv"
user_transformation_csv = "webui/user_transformations.csv"

prompt_styles = styles.StyleDatabase(style_csv, user_style_csv)

prompt_transformations = transformations.TransformationDatabase(transformation_csv, user_transformation_csv)

#prompt_styles = styles.StyleDatabase("webui/styles.csv", "webui/user_styles.csv")
#prompt_transformations = transformations.TransformationDatabase("webui/transformations.csv", "webui/user_transformations.csv")

cancel_process = False

autolaunch = False

global save_log_lines
save_log_lines = 100
if len(sys.argv) > 1:
    autolaunch = "-autolaunch" in sys.argv

scroll_style = """
<style>
    .scrollable {
        max-height: 300px;
        overflow-y: scroll;
        white-space: pre-wrap;
    }
</style>
"""

bark_console_style = """
.bark_console {
font: 1.3rem Inconsolata, monospace;
  white-space: pre;
  padding: 0;
  border: 2px dashed orange;
  border-radius: 3px;
  max-height: 500px; 
  overflow-y: scroll; 
  font-size: 90%;
  overflow-x: hidden;
  }



 
 

   #styles_row  button {
display: flex;
width: 2em;   
     align-self: end;
     margin: 8px 13px 0px 0px;
   }


  #styles_row div .wrap .wrap-inner,  #styles_row div.panel {
padding: 0px;
   margin: 0px;
  min-height: 34px;

 }

 #styles_row div.form {
  border: none;
   position: absolute;
   background: none;
 }


"""
import functools

def timeout(seconds):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = [None]
            thread = threading.Thread(target=lambda: result.__setitem__(0, func(*args, **kwargs)))
            thread.start()
            thread.join(seconds)
            if thread.is_alive():
                return None
            return result[0]
        return wrapper
    return decorator




# I made a CLI app. This is my solution. I'm not proud of it.
def parse_extra_args(extra_args_str):
    extra_args = extra_args_str.split('--')
    parsed_args = {}
    for arg in extra_args:
        if not arg.strip():
            continue
        key, value = arg.strip().split(' ', 1)
        if value.lower() == 'true':
            value = True
        elif value.lower() == 'false':
            value = False
        else:
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass  # If it's not a number, keep it as a string
        parsed_args[key] = value
    return parsed_args

def generate_audio_long_gradio(input, npz_dropdown, generated_voices, bark_infinity_voices, confused_travolta_mode, stable_mode_interval, seperate_prompts, split_character_goal_length, split_character_max_length, process_text_by_each, in_groups_of_size, group_text_by_counting, split_type_string, prompt_text_prefix, seed, text_splits_only,output_iterations,hoarder_mode, text_temp, waveform_temp, semantic_min_eos_p, output_dir, output_filename, output_format, add_silence_between_segments,  semantic_top_k, semantic_top_p, coarse_top_k, coarse_top_p, specific_npz_file, specific_npz_folder, split_character_jitter, extra_args_str, progress=gr.Progress(track_tqdm=True)):
    print("\n")
    if input == None or len(input) < 4:
        print("\nLooks like you forgot to enter a text prompt.")
        text_splits_only = True


    trim_logs()
    global cancel_process

    progress(0, desc="Starting...")
    waiting = 0
    while api.gradio_try_to_cancel and not api.done_cancelling:
        waiting += 1
        print("Waiting up to 10s current generation to finish before starting another...")
        progress(waiting, desc="Waiting up to 10s current generation to finish before starting another...")
        if waiting > 10:
            print("Everything might be okay, but something didn't quite cancel properly so restart if things are weird.")
            break
        time.sleep(1)

    if api.gradio_try_to_cancel and api.done_cancelling:
        cleanup_after_cancel()
        api.gradio_try_to_cancel = False
        api.done_cancelling = False
        cancel_process = False
    
    if api.done_cancelling:
        print("Shouldn't happen, just return for now.")
        return

    
    #print(locals())
    kwargs = {}
    kwargs["text_prompt"] = input
    
    # I must have screwed up why are these values so messed up
    if npz_dropdown != '' and npz_dropdown is not None:
        if len(npz_dropdown.strip()) > 6: kwargs["history_prompt"] = npz_dropdown


    if bark_infinity_voices != '' and bark_infinity_voices is not None:
        if len(bark_infinity_voices.strip()) > 6: kwargs["bark_infinity_voices"] = bark_infinity_voices

    if generated_voices != '' and generated_voices is not None:
        if len(generated_voices.strip()) > 6: kwargs["history_prompt"] = generated_voices




    if specific_npz_file != '' and specific_npz_file is not None:
        kwargs["history_prompt"] = specific_npz_file


    if specific_npz_folder != '' and specific_npz_folder is not None:
        kwargs["specific_npz_folder"] = specific_npz_folder

    kwargs["confused_travolta_mode"] = confused_travolta_mode
    kwargs["split_character_goal_length"] = int(split_character_goal_length)
    kwargs["split_character_max_length"] = int(split_character_max_length)

    if split_character_jitter != '' and split_character_jitter is not None:
        kwargs["split_character_jitter"] = float(split_character_jitter)




    if process_text_by_each is not None and process_text_by_each != '':
        kwargs["process_text_by_each"] = process_text_by_each

    if in_groups_of_size is not None:
        kwargs["in_groups_of_size"] = int(in_groups_of_size)
    
    if group_text_by_counting is not None and group_text_by_counting != '':
        kwargs["group_text_by_counting"] = group_text_by_counting

    if split_type_string is not None and split_type_string != '':
        kwargs["split_type_string"] = split_type_string

    if prompt_text_prefix is not None and prompt_text_prefix != '':
        kwargs["prompt_text_prefix"] = prompt_text_prefix


    
    if seed != '' and seed is not None and seed > 0 or seed < 0:
        # because i moved iterations to Gradio, we can't just pass the seed or
        # it will be reset for iteration.
        # for now, let's set it manually
        #kwargs["single_starting_seed"] = int(seed)
        custom_seed = int(seed)
        api.set_seed(custom_seed) # will also let them renable with -1

    if stable_mode_interval != '' and stable_mode_interval is not None:
        if stable_mode_interval == 'Continuous':
            kwargs["stable_mode_interval"] = 0
        elif stable_mode_interval == 'Stable':
            kwargs["stable_mode_interval"] = 1
        elif stable_mode_interval == 'Stable-2':
            kwargs["stable_mode_interval"] = 2
        elif stable_mode_interval == 'Stable-3':
            kwargs["stable_mode_interval"] = 3
        elif stable_mode_interval == 'Stable-4':
            kwargs["stable_mode_interval"] = 4
        elif stable_mode_interval == 'Stable-5':
            kwargs["stable_mode_interval"] = 5
        else:
            kwargs["stable_mode_interval"] = int(stable_mode_interval)

    if text_splits_only != '' and text_splits_only is not None:
        kwargs["text_splits_only"] = text_splits_only


    if seperate_prompts != '' and seperate_prompts is not None:
        kwargs["seperate_prompts"] = seperate_prompts

    if hoarder_mode != '' and hoarder_mode is not None:
        kwargs["hoarder_mode"] = hoarder_mode

    if semantic_top_k is not None and semantic_top_k != '' and semantic_top_k > 0:
        kwargs["semantic_top_k"] = int(semantic_top_k)
    
    if semantic_top_p is not None and semantic_top_p != '' and semantic_top_p > 0:
        kwargs["semantic_top_p"] = float(semantic_top_p)
    
    if coarse_top_k is not None and coarse_top_k != '' and coarse_top_k > 0:
        kwargs["coarse_top_k"] = int(coarse_top_k)
    
    if coarse_top_p is not None and coarse_top_p != '' and coarse_top_p > 0:
        kwargs["coarse_top_p"] = float(coarse_top_p)
    



    if output_dir is not None and output_dir != '':
        kwargs["output_dir"] = output_dir

    if output_filename is not None and output_filename != '':
        kwargs["output_filename"] = output_filename

    if output_format is not None and output_format != '':
        kwargs["output_format"] = output_format

    #this is obviously got to be the wrong way to do this


    
    if text_temp is not None and text_temp != '':
        kwargs["text_temp"] = float(text_temp)

    if waveform_temp is not None and waveform_temp != '':
        kwargs["waveform_temp"] = float(waveform_temp)

    if semantic_min_eos_p is not None and semantic_min_eos_p != '':
        kwargs["semantic_min_eos_p"] = float(semantic_min_eos_p)

    if add_silence_between_segments is not None and add_silence_between_segments != '':
        kwargs["add_silence_between_segments"] = float(add_silence_between_segments)




   

    # i need to read the docs
    kwargs.update(parse_extra_args(extra_args_str))


    using_these_params = kwargs.copy()
    using_these_params["text_prompt"] = f"{input[:10]}... {len(input)} chars"
    print(f"Using these params: {using_these_params}")



    if output_iterations is not None and output_iterations != '':
        output_iterations = int(output_iterations)
    else:
        output_iterations = 1
        
    if (text_splits_only):
        output_iterations = 1
    full_generation_segments, audio_arr_segments, final_filename_will_be = None,None,None
   

    kwargs["output_iterations"] = output_iterations

    npz_files = None
    if specific_npz_folder is not None and specific_npz_folder != '':
        npz_files = [f for f in os.listdir(specific_npz_folder) if f.endswith(".npz")]
        npz_files.sort()
        if len(npz_files) == 0:
            print(f"Found no npz files in {specific_npz_folder}")
        else:
            total_iterations = kwargs["output_iterations"] * len(npz_files)
 
            print(f"Found {len(npz_files)} npz files in {specific_npz_folder} so will generate {total_iterations} total outputs")
        
    if npz_files is not None and len(npz_files) > 0:

        for i, npz_file in enumerate(npz_files):
            print(f"Using npz file {i+1} of {len(npz_files)}: {npz_file}")
            kwargs["history_prompt"] = os.path.join(specific_npz_folder, npz_file)
            
            for iteration in range(1,output_iterations + 1):
                text_prompt = kwargs.get("text_prompt")
                if output_iterations > 1:
                    if iteration == 1:
                        print("  ", text_prompt)

                kwargs["current_iteration"] = iteration
                progress(iteration, desc=f"Iteration: {iteration}/{output_iterations}...", total=output_iterations)

                full_generation_segments, audio_arr_segments, final_filename_will_be = api.generate_audio_long_from_gradio(**kwargs)

                if cancel_process:
                    return final_filename_will_be
            if kwargs.get('text_splits_only', False):
                final_filename_will_be = "bark_infinity/assets/split_the_text.wav"
        return final_filename_will_be
    else:
        for iteration in range(1,output_iterations + 1):
            text_prompt = kwargs.get("text_prompt")
            if output_iterations > 1:
                if iteration == 1:
                    print("  ", text_prompt)

            kwargs["current_iteration"] = iteration
            progress(iteration, desc=f"Iteration: {iteration}/{output_iterations}...", total=output_iterations)

            full_generation_segments, audio_arr_segments, final_filename_will_be = api.generate_audio_long_from_gradio(**kwargs)

            if cancel_process:
                return final_filename_will_be
        if kwargs.get('text_splits_only', False):
            final_filename_will_be = "bark_infinity/assets/split_the_text.wav"
        return final_filename_will_be

def create_npz_dropdown_dir(directories, label):
    npz_files_by_subfolder = defaultdict(list)
    for directory in directories:
        
        for npz_file in glob.glob(os.path.join(directory, '**', '*.npz'), recursive=True):
            subfolder = os.path.dirname(npz_file)
            npz_files_by_subfolder[subfolder].append(npz_file)
    
    sorted_npz_files = []
    for subfolder in sorted(npz_files_by_subfolder.keys()):
        sorted_npz_files.extend(sorted(npz_files_by_subfolder[subfolder]))
    
    npz_dropdown = gr.Dropdown(sorted_npz_files, label=label, allow_custom_value=True)
    return npz_dropdown

def create_npz_dropdown(directory, label, info="", allow_custom_value=False):
    npz_files_by_subfolder = defaultdict(list)

        
    for npz_file in glob.glob(os.path.join(directory, '**', '*.npz'), recursive=True):
        subfolder = os.path.dirname(npz_file)
        npz_files_by_subfolder[subfolder].append(npz_file)
    
    sorted_npz_files = []
    for subfolder in sorted(npz_files_by_subfolder.keys()):
        sorted_npz_files.extend(sorted(npz_files_by_subfolder[subfolder]))
    
    #npz_dropdown = gr.Dropdown(sorted_npz_files, label=label, info=info, allow_custom_value=allow_custom_value)
    npz_dropdown = gr.Dropdown(sorted_npz_files, label=label, info=info, allow_custom_value=True)

    return npz_dropdown



directories = config.VALID_HISTORY_PROMPT_DIRS

outputs_dirs = ["bark_samples/"]

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return False


sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', newline='', line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', newline='', line_buffering=True)

sys.stdout = Logger("gradio_terminal_ouput.log")
def test(x):

    #print("This is a test")
    #print(f"Your function is running with input {x}...")

    return

def read_logs():
    sys.stdout.flush()
    with open("gradio_terminal_ouput.log", "r", encoding="utf-8") as f:
        return f.read()
    

model_options = [
    ('text_use_gpu', {'value': True, 'type': bool, 'help': "Load the text model on the GPU."}),
    ('text_use_small', {'value': False, 'type': bool, 'help': "Use a smaller/faster text model."}),
    ('coarse_use_gpu', {'value': True, 'type': bool, 'help': "Load the coarse model on the GPU."}),
    ('coarse_use_small', {'value': False, 'type': bool, 'help': "Use a smaller/faster coarse model."}),
    ('fine_use_gpu', {'value': True, 'type': bool, 'help': "Load the fine model on the GPU."}),
    ('fine_use_small', {'value': False, 'type': bool, 'help': "Use a smaller/faster fine model."}),
    ('codec_use_gpu', {'value': True, 'type': bool, 'help': "Load the codec model on the GPU."}),
    ('force_reload', {'value': False, 'type': bool, 'help': "Force the models to be downloaded again."}),
]

def preload_models_gradio(text_use_gpu, text_use_small, coarse_use_gpu, coarse_use_small, fine_use_gpu, fine_use_small, codec_use_gpu, force_reload):
    print("Preloading models...")
    generation.preload_models(
        text_use_gpu=text_use_gpu,
        text_use_small=text_use_small,
        coarse_use_gpu=coarse_use_gpu,
        coarse_use_small=coarse_use_small,
        fine_use_gpu=fine_use_gpu,
        fine_use_small=fine_use_small,
        codec_use_gpu=codec_use_gpu,
        force_reload=force_reload,
    )


def cleanup_after_cancel():

    global cancel_process

    # put all the models on the right device
    generation.preload_models(

        force_reload=True,
    )
    print("Fixing models...")

def try_to_cancel(text_use_gpu, text_use_small, coarse_use_gpu, coarse_use_small, fine_use_gpu, fine_use_small, codec_use_gpu, force_reload):

    global cancel_process
    cancel_process = True
    api.gradio_try_to_cancel = True
    api.done_cancelling = False
    print("Trying to cancel...")

def generate_speaker_variations(variation_path, variation_count):


    # I need to actually read how Gradio is supposed to work... why is this a float?
    if variation_count is not None and variation_count != '':
        variation_count = int(variation_count)
        print(f"Generating {variation_count} for speakers {variation_path}...")
        
        #should still link this as a lighter option
        #api.render_npz_samples(npz_directory=variation_path,gen_minor_variants=variation_count)
        
        
        api.doctor_random_speaker_surgery(variation_path, variation_count)
    return

def generate_sample_audio(sample_gen_path):
    print("Generating sample audio...")
    api.render_npz_samples(npz_directory=sample_gen_path)
    return

def sent_bark_envs(env_config_group,loglevel, save_log_lines_number, text_use_gpu, text_use_small, coarse_use_gpu, coarse_use_small, fine_use_gpu, fine_use_small, codec_use_gpu, force_reload):



    generation.OFFLOAD_CPU = "OFFLOAD_CPU" in env_config_group
    generation.USE_SMALL_MODELS = "USE_SMALL_MODELS" in env_config_group
    generation.GLOBAL_ENABLE_MPS = "GLOBAL_ENABLE_MPS" in env_config_group

    print(f"Setting these envs: OFFLOAD_CPU={generation.OFFLOAD_CPU}, USE_SMALL_MODELS={generation.USE_SMALL_MODELS}, GLOBAL_ENABLE_MPS={generation.GLOBAL_ENABLE_MPS}")

    if loglevel is not None and loglevel != '':
        print(f"Setting log level to {loglevel}")
        logger.setLevel(loglevel)

    global save_log_lines
    save_log_lines = save_log_lines_number

    preload_models_gradio(text_use_gpu, text_use_small, coarse_use_gpu, coarse_use_small, fine_use_gpu, fine_use_small, codec_use_gpu, force_reload)

def set_gradio_options(save_log_lines_number):
    global save_log_lines
    save_log_lines = save_log_lines_number


    generation.OFFLOAD_CPU = OFFLOAD_CPU
    generation.USE_SMALL_MODELS = USE_SMALL_MODELS
    generation.GLOBAL_ENABLE_MPS = GLOBAL_ENABLE_MPS








def generate_gradio_widgets(options):
    widgets = []
    for option_name, option_info in options:
        if option_info['type'] == bool:
            checkbox = gr.Checkbox(label=option_name, value=option_info['value'], info=option_info['help'])
            widgets.append(checkbox)
    return widgets
generated_widgets = generate_gradio_widgets(model_options)


def format_defaults(defaults):
    formatted_text = ""
    for group_name, arguments in defaults.items():
        formatted_text += f"{group_name}:\n"
        for key, arg in arguments:
            formatted_text += f"  {key}:\n"
            formatted_text += f"    Type: {arg['type'].__name__}\n"
            formatted_text += f"    Default: {arg['value']}\n"
            formatted_text += f"    Help: {arg['help']}\n"
            if 'choices' in arg:
                formatted_text += f"    Choices: {', '.join(map(str, arg['choices']))}\n"
            formatted_text += "\n"
    return formatted_text

formatted_defaults = format_defaults(config.DEFAULTS)


# When using Unicode with Python, replace "+" with "000" from the Unicode. And then prefix the Unicode with "\".
# Using constants for these since the variation selector isn't visible.
# Important that they exactly match script.js for tooltip to work.
random_symbol = '\U0001f3b2\ufe0f'  # 🎲️
reuse_symbol = '\u267b\ufe0f'  # ♻️
paste_symbol = '\u2199\ufe0f'  # ↙
refresh_symbol = '\U0001f504'  # 🔄
save_style_symbol = '\U0001f4be'  # 💾
apply_style_symbol = '\U0001f4cb'  # 📋
clear_prompt_symbol = '\U0001f5d1\ufe0f'  # 🗑️
extra_networks_symbol = '\U0001F3B4'  # 🎴
switch_values_symbol = '\U000021C5' # ⇅
restore_progress_symbol = '\U0001F300' # 🌀

text_transformation_symbol = '\U00002728' # ✨
apply_style_symbol = '\U0001F3A8' # 🎨



def create_refresh_button(refresh_component, refresh_method, refreshed_args, elem_id):

    def refresh():
        refresh_method()
        args = refreshed_args() if callable(refreshed_args) else refreshed_args

        for k, v in args.items():
            setattr(refresh_component, k, v)

        return gr.update(**(args or {}))

    refresh_button = ToolButton(value=refresh_symbol, elem_id=elem_id)
    refresh_button.click(
        fn=refresh,
        inputs=[],
        outputs=[refresh_component]
    )
    return refresh_button

def apply_styles(prompt, styles):
    prompt = prompt_styles.apply_styles_to_prompt(prompt, styles)

    return [gr.Textbox.update(value=prompt), gr.Dropdown.update(value=[])]

def apply_transformations(prompt, styles):
    prompt = prompt_transformations.apply_transformations_to_prompt(prompt, styles)

    return [gr.Textbox.update(value=prompt), gr.Dropdown.update(value=[])]


def trim_logs():

    global save_log_lines
    print(f"Trimming logs to {save_log_lines} lines...")
    save_log_lines = int(save_log_lines)

    if save_log_lines < 0:
        return
    
    with open("gradio_terminal_ouput.log", "r") as f:
        lines = f.readlines()

    if save_log_lines > 0 and len(lines) > save_log_lines:
        lines = lines[-save_log_lines:]

    with open("gradio_terminal_ouput.log", "w") as f:
        f.writelines(lines)

with gr.Blocks(theme=default_theme,css=bark_console_style) as demo:
    gr.Markdown(
        """
    # 🐶 Bark Infinity - Text to Audio For Any Universe 🌌 </a>

    Feedback, feature requests, far too many emojis: <a href="https://github.com/JonathanFly/bark">https://github.com/JonathanFly/bark</a>
    """
    )


    with gr.Row():
        with gr.Column(variant="panel", scale=0.5):
            gr.Markdown("### 🐶 Main Bark Input")


            with gr.Row(elem_id=f"text_row"):
                input = gr.TextArea(placeholder="Text Prompt", label="Main Text Prompt", info="The main text goes here. It can be as long as you want. You will see how the text will be split into smaller chunks on the right.")
            
            
            with gr.Row(elem_id=f"styles_row"):
 
           
                with gr.Column(variant="panel", scale=0.5):
                    prompt_styles_dropdown = gr.Dropdown(label=f"Text Snippets", info=f"Add your own! {user_style_csv}", elem_id=f"styles", choices=[k for k, v in prompt_styles.styles.items()], value=[], multiselect=True)
                    #create_refresh_button(prompt_styles_dropdown, prompt_styles.reload, lambda: {"choices": [k for k, v in prompt_styles.styles.items()]}, f"refresh_styles")
                    prompt_style_apply = ToolButton(value=apply_style_symbol, elem_id=f"style_apply")
                    #save_style = ToolButton(value=save_style_symbol, elem_id=f"style_create")
                with gr.Column(variant="panel", scale=0.5):
                    prompt_transformations_dropdown = gr.Dropdown(label=f"Text Transformations", info=f"Add your own! {user_transformation_csv}", elem_id=f"transformations", choices=[k for k, v in prompt_transformations.transformations.items()], value=[], multiselect=True)
                    #create_refresh_button(prompt_styles_dropdown, prompt_styles.reload, lambda: {"choices": [k for k, v in prompt_styles.styles.items()]}, f"refresh_styles")
                    prompt_transformations_apply = ToolButton(value=text_transformation_symbol, elem_id=f"transformation_apply")
                    #save_style = ToolButton(value=save_style_symbol, elem_id=f"style_create")
            prompt_style_apply.click(
                fn=apply_styles,
                inputs=[input, prompt_styles_dropdown],
                outputs=[input, prompt_styles_dropdown],
            )

            prompt_transformations_apply.click(
                fn=apply_transformations,
                inputs=[input, prompt_transformations_dropdown],
                outputs=[input, prompt_transformations_dropdown],
            )
            



        with gr.Column(variant="panel", scale=0.5):
            gr.Markdown("""### ⌨️ Bark 'Console'
            This began as a command line, and real time updates are still logged here. The log trims old entries unless you set other options.""")
        
            output = gr.HTML(elem_classes ="bark_console", interactive=True)

            
            def clear_logs():
                with open("gradio_terminal_ouput.log", "w") as f:
                    f.write("")


            clear_button = gr.Button("Clear The Console")
            clear_button.click(clear_logs)
         


    with gr.Tab("Main Options"):

        with gr.Row():
            with gr.Column(variant="panel",scale=0.75):
                m("## 🧑‍🎤 Pick a Speaker")
                with gr.Tab("🌱🎙️ Create New Speaker"):
                    m("This will create a new speaker speaker.npz file. The voice is random but the text matters. Like GPT, <a href='https://twitter.com/jonathanfly/status/1649637372949155840'>try to imagine a context</a> where the thing or voice you’re looking for would naturally follow. If you're exploring, consider checking the 💎💎 'Save Everything' checkbox so long audio clips can produce multiple speakers.")
                    m("If you want a random speaker make sure to clear out all the dropdown menus. UI is work in progress.")
                with gr.Tab("🧑‍🎤 Build In Speakers"):
                    npz_dropdown = create_npz_dropdown("bark/assets/prompts/", label="Speaker", info="These are speakers provided by Suno-ai, in many languages. The v2 ones are good for a basic clear voice.")
                    gr.Markdown("🌌🧙 Bark Infinity Speaker")
                    gr.Markdown("""Speakers I accidentally left in in when I first made this. They will get better <a href='https://github.com/JonathanFly/bark/discussions/47'>contribute</a> to contribute.""")
                    bark_infinity_voices = create_npz_dropdown("bark_infinity/assets/prompts/", label="Speaker")
                with gr.Tab("👩‍🎤🎙️ Your Custom Speaker"):

                    gr.Markdown("""Use your own Speaker .npz file. This looks in directory "custom_speakers/" but you can type in a custom full path to a speaker.npz file as well.""")

                    generated_voices = create_npz_dropdown("custom_speakers/", label="Speaker", allow_custom_value = True, info="You can enter a custom file path here as well.")

                with gr.Tab("👩‍🎤🎙️ Folder Processing"):
                    gr.Markdown("""For each file""")
                    specific_npz_file = gr.Textbox(label="direct path to .npz", value="")
                    specific_npz_folder = gr.Textbox(label="folder of npz files", value="")
   

            with gr.Column(variant="panel",scale=0.25):
                m("## ...")
                #m("Chosen Speaker")
   



                
        with gr.Row():

            with gr.Column(variant="panel", scale=0.25):
                
                m("## ✂️ Split Up Long Text")

                with gr.Tab("Simple"):
                    m("Try to aim about 10s per audio clip. It's fine to leave these on defaults. ")
                    split_character_goal_length = gr.Slider(label="Try for this many characters in each", value=165, maximum=500, step=1)
                    split_character_max_length = gr.Slider(label="But never go higher than this many", value=205, maximum=500, step=1)
                
                with gr.Tab("Fancy"):
                    prompt_text_prefix = gr.Textbox(label="Put this text in front of every prompt, after splitting.", value="")
                    split_character_jitter = gr.Slider(label="Randomize character splits by this much", info="If you're generating a lot of iterations you might try randomizing the splits a bit with this.", value=0, maximum=500, step=1)
                    m("Below is mostly placeholder. But these old functions still sort of work:")
                    m("For example for song lyrics, in the below 3 boxes pick: `line` then `4` then `line` this will split the text in groups of 4 lines each.")
                    process_text_by_each = gr.Dropdown(['word', 'line', 'sentence', 'char', 'string', 'random', 'regex'], label="Process the text in chunks of:", value=None)
                    group_text_by_counting = gr.Dropdown(['word', 'line', 'sentence', 'char', 'string', 'random', 'regex'], label="Group the text by counting:", value=None)
                    in_groups_of_size = gr.Slider(label="And start a new audio clip with you have this many:", minimum=1, maximum=50, step=1, value=None)
                    
                    split_type_string = gr.Textbox(label="(Optional String for string or regex.)", value="")

                text_splits_only = gr.Checkbox(label="🗺️✂️ No audio, just show me text splits.", value=False)


            with gr.Column(variant="panel", scale=0.25):

                m("## 🌉Connect Audio Segments")
                with gr.Tab("Simple"):
                    m("### How to Join Clips")
                    
                    stable_mode_interval = gr.Dropdown(["Continuous", "Stable", "Stable-2","Stable-3","Stable-4","Stable-5"], label="How to Join Clips", info="", value="Stable")

                    m(""" - *Stable* for reliable long clips.
                    - *Continuous* for voices that keep evolving.
                    - *Stable-2* through *Stable-5* will reset back after that many segments.
                    - (coming soon, *Magic* where it just works...)""")




            
                with gr.Tab("Fancy"):


                    semantic_min_eos_p = gr.Slider(label="How likely the clip will end.", minimum=0.0, maximum=1.0, value=0.2, interactive=True, info="Try 0.10 or 0.05 if you're getting extra words.")


                    add_silence_between_segments = gr.Slider(label="Add Silence", minimum=0.0, maximum=5.0, value=0.0, interactive=True, info="Try 0.25 if using 'Stable' mode to space it out a bit.")
                
                    m("### Enlarge or clip histories. Not in this version yet.")
                    history_prompt_semantic_weight = gr.Slider(label="History Prompt Semantic Weight", minimum=0.0, maximum=2.0, value = 1.0, interactive = True)
                    history_prompt_coarse_weight = gr.Slider(label="History Prompt Coarse Weight", minimum=0.0, maximum=2.0, value = 1.0, interactive = True)
                    history_prompt_fine_weight = gr.Slider(label="History Prompt Fine Weight", minimum=0.0, maximum=2.0, value = 1.0, interactive = True)

                    prev_semantic_weight = gr.Slider(label="Prev Semantic Weight", minimum=0.0, maximum=2.0, value = 1.0, interactive = True)
                    prev_coarse_weight = gr.Slider(label="Prev Coarse Weight", minimum=0.0, maximum=2.0, value = 1.0, interactive = True)
                    prev_fine_weight = gr.Slider(label="Prev Fine Weight", minimum=0.0, maximum=2.0, value = 1.0, interactive = True)


                with gr.Tab("Don't Connect Them"):
                    m("### Split the text, but treat each segment like it's own prompt.")
                    m("Good for discovering speakers.")
                    seperate_prompts = gr.Checkbox(label="Seperate Prompts", value=False, interactive=True, visible=True)


            with gr.Column(variant="panel", scale=0.25):

                m("## Generate")

                with gr.Tab("Simple"):
  
                    m("### higher = more diversity, lower more conservative")
                    text_temp = gr.Slider(label="text temperature: ", minimum=0.0, maximum=1.0, value = 0.70, interactive = True)
                    waveform_temp = gr.Slider(label="wave temperature: ", minimum=0.0, maximum=1.0, value=0.70, interactive = True)

       

                with gr.Tab("Fancy"):
                    seed = gr.Number(label="Random SEED: 0 for no seed. Set -1 to undo.", info="(Bark runs a lot slower when using a seed.)", value=0)
                    m("""## These options should in theory have a decent impact
                    But it's been hard to say for sure
                    `top_k` 50 is typical, 
                    `top_p` 0.90 or 0.95,
                    lower top_p less diverse
                    you don't have to use both.""")
                    semantic_top_k = gr.Slider(label="semantic_top_k", value=50, minimum=0, maximum=200, step=1)
                    semantic_top_p = gr.Slider(label="semantic_top_p", value=0.95, minimum=0.0, maximum=1.0)
                    coarse_top_k = gr.Slider(label="coarse_top_k", value=50, minimum=0, maximum=200, step=1)
                    coarse_top_p = gr.Slider(label="coarse_top_p", value=0.92, minimum=0.0, maximum=1.0)
                    m("""## Anecdotally:
                     1. Improves hit rate in random voices.
                     2. Might reduce the weirder outputs.
                     3. Could using same sampling parameters that made original speaker matter?
                     4. I'm harcoding these to on because people probably like more reliabilitiy.
                     5. (Seting them to 0 will disable them.""")

                with gr.Tab("Fun"):       
                    confused_travolta_mode = gr.Checkbox(label="🕺🕺 Confused Travolta Mode", info="Make Bark to keep talking even when it is finished with your prompt.", value=False)
            with gr.Column(variant="panel", scale=0.25):
                m("## Output")
                with gr.Tab("Simple"):
                    hoarder_mode = gr.Checkbox(label="💎💎Save Every NPZ", value=False)
                    output_dir = gr.Textbox(label="Output directory", value="bark_samples/")
                    
                    output_iterations = gr.Slider(label="Repeat This Many Times", step=1, value=1, minimum=1, maximum=1000)
                with gr.Tab("Fancy"):
                    output_filename = gr.Textbox(label="Output filename", value="", info="Use prompt, speaker, and date if left blank.")
                    output_format = gr.Dropdown(['wav','mp3', 'ogg', 'flac', 'mp4'], value='wav', label="Audio File Output Format", info="(You can re-render wavs if you save .npzs)")





            

    with gr.Tab("Model Options, Configs, Setup"):
        with gr.Row():
            gr.Markdown("Preloading is a little bit faster if you have enough GPU memory, but the difference is actutally pretty small. You can still use the larger models just fine without preloading them, they get swapped out to CPU by default in this version.")
        with gr.Row():
            with gr.Column(scale=.25, variant="panel"):
                gr.Markdown("## Model Options")
                gr.Markdown("You can probably use the big models, even on low GPU memory.")
                model_checkboxes = generate_gradio_widgets(model_options)
                model_button = gr.Button("Preload These Models")
                model_button.click(preload_models_gradio, inputs=model_checkboxes) 

            with gr.Column(scale=.75, variant="panel"):
                gr.Markdown("## System Wide Settings")
                gr.Markdown("If you have 10GB of VRAM")
                m("Then click 'Set these parameters' with OFFLOAD_CPU unchecked. If you already preloaded models, preload again with force_reload=True.")
                env_config_vars = ["OFFLOAD_CPU", "USE_SMALL_MODELS", "GLOBAL_ENABLE_MPS"]
                env_config_values = ["True", "False", "False"]
                env_config_group= gr.CheckboxGroup(choices=env_config_vars, value=env_config_values, label="System Wide Config Settings", type="value", interactive=True, visible=True)
                
                loglevel = gr.Dropdown(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], label="Log Level", info="DEBUG = Drown in Text")

                
                save_log_lines_number = gr.Number(label="When you click Generate, clear all but this many lines from the console",
                                            value=100)
                
                env_button = gr.Button("Set These Config Options")
                env_input_list = [env_config_group] + [loglevel, save_log_lines_number] + model_checkboxes

                env_button.click(sent_bark_envs, inputs=env_input_list)

 
    with gr.Tab("👨🏻‍⚕️🧬Speaker Surgery Center"):
        with gr.Row():
            with gr.Column(scale=.25):
                m("### 🚑 Lost track of what your .npz files sound like?")
                m("Quickly generate a sample audio clip for each speaker file in a directory.")
                sample_gen_path = gr.Textbox(label="Sample Directory", value="bark/assets/prompts/v2")
                sample_gen_button = gr.Button("Gen Voice Samples", variant="primary")
                sample_gen_button.click(generate_sample_audio, inputs=sample_gen_path)

            with gr.Column(scale=.50):
                gr.Markdown("### 🏣 Speaker Surgery.")
                gr.Markdown("Have a great voice but something isn't right? Wish you you could fix it? You've come to the right place. First, did you already try making a wide variety of new clips with different prompts and re-saving it? But if that doesn't work, it might be time to call in the doctor.""")
                with gr.Tab("### Doctor RNG 👩🏻‍⚕️🎲"):
                    gr.Markdown("""We've just opened the surgery center and our first hire is a bit questionable. We can't promise to *fix* your troubled .npz.
                    But we *can* close our eyes and slice and dice it up randomly. You'll end up with a lot of versions ofs your original file. Not the most efficient method of medical care, but you know what they say about . Don't worry we have more doctors on the way.""")
                    variation_path = gr.Textbox(label="Speaker NPZ Path", value="bark_samples/myspeakerfile.npz")
                    variation_count = gr.Number(label="How Many Variations", value=10)
                    generate_speaker_variations_button = gr.Button("Generate Voice Variations", variant="primary")
                
                    generate_speaker_variations_button.click(generate_speaker_variations,inputs=[variation_path, variation_count])

                with gr.Tab("### Doctor 🌪️👩🏻‍⚕️"):
                    gr.Markdown("""### Coming soon...""")
 

   
    with gr.Tab("Even More Options"):
        with gr.Row():
                with gr.Column(scale=.33, variant="panel"):
                    m("# You might not have asked for a command line interface in your Gradio app, but it sure beats me making 80 more checkboxes.")
                    m("Some of these options even work. Type them like you would on a command line.")
                    m("```--semantic_top_k 50```")
                    m("```--semantic_min_eos_p 0.05```")
                
                with gr.Column(scale=.33, variant="panel"):
                    m("### 🐍🐍 Raw list of some advanced options that may or may not be implemented or working.")
                    gr.HTML(f"{formatted_defaults}",elem_classes ="bark_console", info=". I cut a lot of these out becaus they were buggy or took too long to try and merge with regular Bark because I don't really understand the stuff I poke at very well.")
                with gr.Column(scale=.33, variant="panel"):

                    extra_args_input = gr.TextArea(lines=15, label="Extra Arguments", elem_classes ="bark_console")

    
    with gr.Tab("🎤 Clone a Voice? 🤷"):
        # Copied from https://github.com/serp-ai/bark-with-voice-clone and https://github.com/C0untFloyd/bark-gui, haven't really got anything useful from it so far.
        with gr.Row():
            with gr.Column(scale=1, variant="panel"):
                gr.Markdown("### 🎤 Clone a Voice? Maybe?")
                gr.Markdown("This code is from https://github.com/serp-ai/bark-with-voice-clone and from https://github.com/C0untFloyd")
                gr.Markdown("The only thing I did was have it spit out a gen multiple variants voices rather than one.")
                gr.Markdown("So far no really good clones, and a large percentage of the .npz files don't load up.")
                gr.Markdown("The successful ones that that did load successfuly were VERY short, like 3 to 5 second wav files.")
                gr.Markdown("They had some resemblence to the original speaker, but the voices were pretty bad.")
                input_audio_filename = gr.Audio(label="Input audio.wav", source="upload", type="filepath")
                transcription_text = gr.Textbox(label="Transcription Text", lines=1, placeholder="Enter Text of your Audio Sample here...")
                initialname = "ClonedVoice"
                #inputAudioFilename = gr.Textbox(label="Filename of Input Audio", lines=1, placeholder="audio.wav")
                output_voice = gr.Textbox(label="Filename of trained Voice", lines=1, placeholder=initialname, value=initialname)
                clone_voice_button = gr.Button("Create Voice")
                dummy = gr.Text(label="Progress")

                clone_voice_button.click(clone_voice, inputs=[input_audio_filename, transcription_text, output_voice], outputs=dummy)
    

    with gr.Row(): 
        with gr.Column(scale=1):
            generate_button = gr.Button("Generate!", variant="primary")
        

        with gr.Column(scale=1):
            
            cancel_button = gr.Button("Cancel. It works, just hit it once and give it few seconds.", label="", variant="stop")

    with gr.Row():


            audio_output = gr.Audio(label="Bark Sample", type="filepath")






    

    theme_selector = gr.Radio(
        ["Base", "Default", "Monochrome", "Soft", "Glass"],
        value="Base",
        label="",
    )
    theme_selector.change(
        None,
        theme_selector,
        None,
        _js=f"""
        (theme) => {{
            if (!document.querySelector('.theme-css')) {{
                var theme_elem = document.createElement('style');
                theme_elem.classList.add('theme-css');
                document.head.appendChild(theme_elem);

                var link_elem = document.createElement('link');
                link_elem.classList.add('link-css');
                link_elem.rel = 'stylesheet';
                document.head.appendChild(link_elem);
            }} else {{
                var theme_elem = document.querySelector('.theme-css');
                var link_elem = document.querySelector('.link-css');
            }}
            if (theme == "Base") {{
                var theme_css = `{base_theme._get_theme_css()}`;
                var link_css = `{base_theme._stylesheets[0]}`;
            }} else if (theme == "Default") {{
                var theme_css = `{default_theme._get_theme_css()}`;
                var link_css = `{default_theme._stylesheets[0]}`;
            }} else if (theme == "Monochrome") {{
                var theme_css = `{monochrome_theme._get_theme_css()}`;
                var link_css = `{monochrome_theme._stylesheets[0]}`;
            }} else if (theme == "Soft") {{
                var theme_css = `{soft_theme._get_theme_css()}`;
                var link_css = `{soft_theme._stylesheets[0]}`;
            }} else if (theme == "Glass") {{
                var theme_css = `{glass_theme._get_theme_css()}`;
                var link_css = `{glass_theme._stylesheets[0]}`;
            }}
            theme_elem.innerHTML = theme_css;
            link_elem.href = link_css;
        }}
    """,
    )


    
    generate_event = generate_button.click(generate_audio_long_gradio,inputs=[input, npz_dropdown, generated_voices,bark_infinity_voices, confused_travolta_mode,stable_mode_interval,seperate_prompts, split_character_goal_length,split_character_max_length, process_text_by_each, in_groups_of_size, group_text_by_counting, split_type_string, prompt_text_prefix, seed, text_splits_only, output_iterations, hoarder_mode, text_temp, waveform_temp,semantic_min_eos_p, output_dir, output_filename, output_format, add_silence_between_segments, semantic_top_k, semantic_top_p, coarse_top_k, coarse_top_p, specific_npz_file, specific_npz_folder, split_character_jitter, extra_args_input], outputs=[audio_output])

    
    cancel_button.click(fn=try_to_cancel, inputs=model_checkboxes, outputs=None, cancels=[generate_event])

    

    logs = gr.HTML()
    # this is crazy right? nobody should have to do this to show text output to Gradio?
    demo.load(read_logs, None, output, every=1)
    

 

#demo.queue(concurrency_count=2, max_size=10)
demo.queue()

# demo.launch(inbrowser=autolaunch)
demo.launch(inbrowser=autolaunch)


