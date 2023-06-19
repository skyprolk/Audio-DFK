# 🚀 BARK INFINITY, Voices are Just Sounds. 🎶 🌈✨🚀 

[![Open Gradio In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1t84qbluQSg7V-YzKit8cD3btmgysT_8V?usp=sharing) Barebone Gradio Running in Google Colab


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Lebdbbq7xOvl9Q430ly6sYrmYoDvlglM?usp=sharing) Basic Colab Notebook

# 🎉 Bark INFINITY Automatic Windows Installer, NVIDIA (CPU update soon) 🎉  


### ⚠️ Note: make sure you fully extract the .zip file before running the .bat files. Check this image if you aren't sure: [install_help.PNG](https://raw.githubusercontent.com/JonathanFly/bark/main/one-click-bark-installer/install_help.PNG)


## Install Prerequisites: 

1. **Just the regular Windows NVIDIA drivers**. You don't need anything else installed ahead of time. Not Pytorch. Nothing with `Cuda` in the name. Not even Python. In fact if you installed anything on your Windows system without using a venv or conda, it may cause a problem.
2. *(Optional But Recommended)* The Windows Terminal https://apps.microsoft.com/store/detail/windows-terminal/9N0DX20HK701 -- Bark still has a lot of text output and it's looks nicer and is easier to read in the Windows Terminal. But you can also use the regular Windows Command Prompt.

## Install Steps

1. Download this zip file: [barkinstall_06_17_23.zip](https://github.com/JonathanFly/bark/raw/main/one-click-bark-installer/barkinstall_06_17_23.zip)
2. Extract the zip file into a directory. Choose a place where Bark will be installed. You will unzip about six small files.
3. Click on `INSTALL_bark_infinity_windows.bat` (you should not need to be administrator)
4. If the install finished with no errors, close that terminal window. Close any other open command line windows as well. 
5. Click `LAUNCH_already_installed_bark_infinity_windows.bat`

## Install Problems

1. If you get a Windows permissions error, I seemed to get it randomly. Just trying again usually fixed it. Not even restarting from scratch, just click on the installer the launch bat.


### Command Line: 
Click `TROUBLESHOOT_bark_setup_manually_by_entering_the_conda_environment.bat`
```
cd bark
python bark_perform.py
python bark_perform.py --help
```
### Trouble Shooting: 
Click `TROUBLESHOOT_bark_setup_manually_by_entering_the_conda_environment.bat`

```
-----Manual Updates-----
Type `conda update -y -n base conda` to update conda.
Type `conda update -y --all --solver=libmamba` to update all packages.
Type `conda clean --all` to free up disk space from unused versions.
Type `ffdl install -U --add-path` to try to fix ffmpeg not not problems.
Type `pip install -r requirements-extra.txt` to try to manually install pip requirements.

Type `conda env update -y -f environment-cuda-installer.yml --prune --solver=libmamba` to update your env manually, if the .yml changed.
Type `cd bark` to enter the bark directory and then `git pull` to update the repo code.
Type `git branch` to view branches and then
Type `git checkout <branch_name>` to switch branches.
(You can try `git checkout bark_amd_directml_test` branch if you have an AMD GPU)

-----Still Not Working?-----
Go ahead and @ me on Bark Official Discord, username "Jonathan Fly" jonathanfly. 
Don't worry about waking me up with a message, my Discord never makes audible alerts.

-----How do I get out of here?-----
Type 'conda deactivate' to exit this environment and go back to normal terminal.
```

![LAUNCH_already_installed_bark_infinity_windows.bat](https://github.com/JonathanFly/bark/assets/163408/fcd91d15-6bee-44c7-8c99-95ca48fbc1d5)

# 🎉 Bark AMD Install Test 🎉

May or not work. I tried to get Bark working in DirectML so AMD Windows users can use it. I don't have an AMD GPU to test myself, but I did work around a lot of Bark DirectML errors. 

To make it work I just replaced any pytorch functions which were causing DirectML errors, with CPU numpy equivalents. I don't know if that's why it's so slow, but with With DirectML in Bark, my 3090 is only about 2x faster than my (very fast) CPU in Bark, so this is really the bare minimum. (It doesn't use up 100% cpu though, at least.) I also couldn't figure out how to free GPU memory in DirectML, so I'm not sure this will work on AMD cards with 8GB of less memory. 

Bark AMD DirectML Instructions.

What is DirectML? 
https://learn.microsoft.com/en-us/windows/ai/directml/gpu-pytorch-windows

Install Miniconda. https://repo.anaconda.com/miniconda/Miniconda3-py310_23.3.1-0-Windows-x86_64.exe

Then go to start menu and start a new "Ananconda Prompt" not regular windows command line 

```
conda update -y conda
conda update -y -n base conda
conda install -y -n base conda-libmamba-solver
conda create --name pydml -y python=3.10
conda activate pydml
```

make sure you see (pydml) in the corner of of your prompt. 
***(pydml) C:\Users\YourName***

```
conda install -y pip git --solver=libmamba
conda update -y --all --solver=libmamba

pip install ffmpeg_downloader
ffdl install -U --add-path
```
Now quit out of the terminal and restart. We need ffmpeg in the path, which means you need to be able to type `ffmpeg -v version` and have it work. If you close and restart, you should be able to do that.

So close the terminal, close all window command lines or terminals to be sure.
Then go back start menu and start a new "Ananaconda Prompt". This should be same you started the install.

```
conda activate pydml
```
make sure you see (pydml) in the corner again. ***(pydml) C:\Users\YourName*** etc.

Now try typing
```
ffmpeg -v version
```

Do you see ffmpeg 6.0? If it doesn't work you can keep going and you can use .wav file outputs, and fix it later.

Now the big conda install command. This could take 5 to 15 minutes, and if you have a slow internet it could even take hours, because it downloads multiple gigabytes. So if looks like it's frozen, let it go. Check your task manager and see if it's downloading.

```
conda install -y pytorch==1.13.1 pygments numpy pandas tensorboard matplotlib tqdm pyyaml boto3 funcy torchvision==0.14.1 torchaudio==0.13.1 cpuonly transformers pydub pathvalidate rich nltk chardet av hydra-core>=1.1 einops scipy num2words pywin32 ffmpeg ffmpeg-python sentencepiece spacy==3.5.2 librosa jsonschema -c pytorch -c conda-forge --solver=libmamba
```
Now that's done a few more things we need, that are not in conda. So we have to use pip.

This is where the instal can go wrong up. **We don't want anything to upgrade either torch or torchaudio to torch 2.0**, and it often happens by accident. (As far I know AMD DirectML Windows only works in Torch 1.13, not 2.0. If anyone knows different let me know!) 

If you somehow end up installing torch 2.0. Try `pip uninstall torch torchaudio` and then redo the big long conda install command (the one with `pytorch==1.13.1` in it). 

```
pip install universal-startfile hydra_colorlog julius soundfile==0.12.1 gradio>=3.35.2 rich_argparse flashy>=0.0.1 ffmpeg_downloader rich_argparse devtools
```


```
pip install encodec flashy>=0.0.1 audiolm_pytorch==1.1.4 demucs --no-dependencies

pip install https://github.com/Sharrnah/fairseq/releases/download/v0.12.4/fairseq-0.12.4-cp310-cp310-win_amd64.whl --no-dependencies
```

And now finally the actual `torch-directml` that has GPU support. I found installing this last seems best, but you could try doing it earlier. 
```
pip install torch-directml==0.1.13.1.dev230413
```
If everything worked, you might be done.
Now we install Bark. And then run one command line test first with bark_perform.py

First set a SUNO_USE_DIRECTML variable. This tells Bark to use DirectML. If this doesn't work you can edit `/bark_infinity/config.py`` and set `SUNO_USE_DIRECTML`` to `True`` in the `DEFAULTS`` section.
```
set SUNO_USE_DIRECTML=1
```

Download Bark:
```
git clone https://github.com/JonathanFly/bark.git
cd bark
```
Change to the AMD Test Version
```
git checkout bark_amd_directml_test
```

Now try running it. Bark has to download all the models the first time it runs, so it might look frozen for awhile. It's another 10 gigs of files. 
```
python bark_perform.py
```
When I tested this install, `bark_perform.py` seemed to freeze at downloading models without making progress. I don't know if was a fluke, but I ran `python bark_webui.py` and it downloaded them fine.

Start the Bark UI
```
python bark_webui.py
```

Things that don't work:
1. Voice Cloning
2. Top_k and top_p
3. Probably more things I haven't tested.

### Start Back UI Later
1. Click Anaconda Prompt in start menu
2. `conda activate pydml`
3. cd bark
4. `python bark_webui.py`
   
### Make it faster? (Note for later, don't try yet)

1. Install MKL exe https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html
```
conda install -y mkl mkl-service mkl_fft libcblas liblapacke liblapack blas-devel mkl-include mkl_random mkl-devel mkl-include ibblas=*=*mkl mkl-static intel-openmp blas=*=*mkl -c intel -c conda-forge --solver=libmamba
```


⚡ Low GPU memory? No problem. CPU offloading. ⚡ Somewhat easy install?

# 🎉 Install Bark Infinity Any OS With Mamba (or Conda) 🎉  


## Mamba Install (Still Works) (out of date, hang on...)

![cpic](https://github.com/JonathanFly/bark/assets/163408/65ad739d-552e-43d2-b937-8c83837bdf8b)

(Mamba is a fast version of conda. They should work the same.)

Pip and conda/mamba are two _different_ ways of installing Bark Infinity. If you use **Mamba** do not install anything. Don't install _pytorch_, do not install anything with 'CUDA' in the same. You don't need to lookup a YouTube tutorial. Just type the commands. The only thing you need installed is the NVIDIA drivers. If you have a mac, use  `environment-cpu.yml` instead of `environment-cpu.yml`

There is one exception, on Windows if you don't have the better Windows Terminal installed, that is a nice to have feature https://apps.microsoft.com/store/detail/windows-terminal/9N0DX20HK701

You don't have to but it may display the output from the bark commands better. When you start **Anaconda Prompt (miniconda3)** you can do it from the new Windows Terminal app, clicking on the down arrow next to the plus, should let you pick **Anaconda Prompt (miniconda3)**

(There is a **requirements-pip.txt** file but I have not tested it currently. This is an alternative install.)

1. [Go here: https://github.com/conda-forge/miniforge#mambaforge](https://docs.conda.io/en/latest/miniconda.html)
2. Download a **Python 3.10 Miniconda3** installer for your OS.  Windows 64-bit, macOS, and Linux probably don't need a guide. 
  a. Install the **Mambaforge** for your OS, not specifically Windows. OSX for OSX etc.
  b. Don't install Mambaforge-pypy3. (It probably works fine, it is just not what I tested.) Install the one above that, just plain **Mambaforge**. Or you can use **Conda**, Mamba should faster but sometimes Conda may be more compatible. 
  
3. Install the **Python 3.10 Miniconda3** exe. Then start the miniforge **'Miniforge Prompt** Terminal which is a new program it installed. You will always use this program for Bark.
   
4. Start **'Miniforge Prompt**  Be careful not to start the regular windows command line. (Unless you installed the new Terminal and know how to switch.) It should say **"Anaconda Prompt (miniconda3)**"

You should see also terminal that says "**(base)**". 

### Do not move forward until you see _(base)_.

5. **Choose the place to install Bark Infinity directory.** You can also just leave it at default. If you make a LOT of audio you think about a place with a lot of space.

When you start **"Anaconda Prompt (miniconda3)"** you will be in a directory, in Windows, probably something like** "C:\Users\YourName"**. Okay to install there. Just remember where you put it. It will be in **/bark.** (If you already had bark-infinity installed and want to update instead of reinstalling, skip to the end.)

6. Type the next commands _exactly_. Hit "Y" for yes where you need to:

## Note for NON-WINDOWS INSTALL
The line 
```pip install fairseq@https://github.com/Sharrnah/fairseq/releases/download/v0.12.4/fairseq-0.12.4-cp310-cp310-win_amd64.whl``` 
is for *Windows Only*. 
If you have Linux or Mac, use `mamba install fairseq` or `pip install fairseq` instead of that line. (If you are on Windows don't change anything, just copy and paste it as you see it.

```
mamba update -y mamba
mamba install -y git
git clone https://github.com/JonathanFly/bark.git
cd bark
mamba env create -f environment-cuda.yml
mamba activate bark-infinity-oneclick
python -m pip install --upgrade pip
pip install --upgrade setuptools 
pip install fairseq@https://github.com/Sharrnah/fairseq/releases/download/v0.12.4/fairseq-0.12.4-cp310-cp310-win_amd64.whl
pip install -r requirements_conda_missing.txt
python bark_perform.py
python bark_webui.py
```

(If you see a warning that "No GPU being used. Careful, inference might be very slow!" after `python bark_perform.py` then something may be wrong, if you have GPU. If you didn't see that then the GPU is working.)

# Start Bark Infinity At A Later Time

To restart later, start **Miniforge Prompt.** Not Regular Prompt. Make sure you see (base) You will type a command to activate **bark-infinity-oneclick** and of base, like this:

```
mamba activate bark-infinity-oneclick
cd bark
python bark_webui.py
```

# Update Bark Infinity 

```
git pull
mamba env update -f environment-cuda.yml --prune
python -m pip install --upgrade pip
pip install --upgrade setuptools 
pip install -r requirements_conda_missing.txt
```

# Pip install. For people who know what they are doing. For this one you are a bit on your own.
```
!git clone https://github.com/JonathanFly/bark.git
%cd bark
!pip install -r requirements-pip.txt
```

## IF you get a strange error during `pip install -r requirements_conda_missing.txt` and are on a Windows computer type this:

```
pip install fairseq@https://github.com/Sharrnah/fairseq/releases/download/v0.12.4/fairseq-0.12.4-cp310-cp310-win_amd64.whl
```

If you are on a mac, you may need to just type `mamba install fairseq` (the windows conda version is too out of date)


I have so much good Bark I need to post at [twitter.com/jonathanfly](https://twitter.com/jonathanfly)


# 🌠 The Past: 🌠

Bark Infinity started as a humble 💻 command line wrapper, a CLI 💬. Built from simple keyword commands, it was a proof of concept 🧪, a glimmer of potential 💡.

# 🌟 The Present: 🌟

Bark Infinity _evolved_ 🧬, expanding across dimensions 🌐. Infinite Length 🎵🔄, Infinite Voices 🔊🌈, and a true high point in human history: [🌍 Infinite Awkwardness 🕺](https://twitter.com/jonathanfly/status/1650001584485552130). But for some people, the time-tested command line interface was not a good fit. Many couldn't even try Bark 😞, struggling with CUDA gods 🌩 and being left with cryptic error messages 🧐 and a chaotic computer 💾. Many people felt very… UN INFINITE. 

# 🔜🚀 The Future: 🚀

🚀 Bark Infinity 🐾 was born in the command line, and Bark Infinity grew within the command line. We live in the era where old fashioned command line applications are wrapped in ✨fancy Gradio Uis🌈 and 🖱️One Click Installers. We all must adapt to a changing world, right? *Or do we?* 

![bark_test_webui](https://user-images.githubusercontent.com/163408/235910939-fa9ae2d6-9a2e-49d2-9646-d07a0793f7b7.PNG)

## 🌟 (OLD NOT UPDATED) README 🌟 __ 

### 1. INFINITY VOICES 🔊🌈
Discover cool new voices and reuse them. Performers, musicians, sound effects, two party dialog scenes. Save and share them. Every audio clip saves a speaker.npz file with the voice. To reuse a voice, move the generated speaker.npz file (named the same as the .wav file) to the "prompts" directory inside "bark" where all the other .npz files are.

🔊 With random celebrity appearances!

(I accidentally left a bunch of voices in the repo, some of them are pretty good. Use --history_prompt 'en_fiery' for the same voice as the audio sample right after this sentence.)

https://user-images.githubusercontent.com/163408/233747981-173b5f03-654e-4a0e-b71b-5d220601fcc7.mp4


### 2. INFINITY LENGTH 🎵🔄
Any length prompt and audio clips. Sometimes the final result is seamless, sometimes it's stable (but usually not both!).

🎵 Now with Slowly Morphing Rick Rolls! Can you even spot the seams in the most earnest Rick Rolls you've ever heard in your life?

https://user-images.githubusercontent.com/163408/233747400-b18411f8-afcb-437d-9288-c54cc2c95e62.mp4

### 🕺 Confused Travolta Mode 🕺
Confused Travolta GIF
![confused_travolta](https://user-images.githubusercontent.com/163408/233747428-c6bf03e2-b3ce-4ce3-a29d-836bf73a4ec2.gif)

Can your text-to-speech model stammer and stall like a student answering a question about a book they didn't read? Bark can. That's the human touch. The *semantic* touch. You can almost feel the awkward silence through the screen.

## 💡 But Wait, There's More: Travolta Mode Isn't Just A Joke 💡

Are you tired of telling your TTS model what to say? Why not take a break and let your TTS model do the work for you. With enough patience and Confused Travolta Mode, Bark can finish your jokes for you. 

https://user-images.githubusercontent.com/163408/233746957-f3bbe25f-c8f0-4570-97b1-1005e1b40cbe.mp4

Truly we live in the future. It might take 50 tries to get a joke and it's probably an accident, but all 49 failures are also *very* amusing so it's a win/win. (That's right, I set a single function flag to False in a Bark and raved about the amazing new feature. Everything here is small potatoes really.)

https://user-images.githubusercontent.com/163408/233746872-cac78447-8e87-49e7-b79b-28ec51264019.mp4



_**BARK INFINITY** is possible because Bark is such an amazingly simple and powerful model that even I could poke around easily._

_For music, I recommend using the --split_by_lines and making sure you use a multiline string as input. You'll generally get better results if you manually split your text, which I neglected to provide an easy way to do because I stayed too late listening to 100 different Bark versions of a scene an Andor and failed Why was 6 afraid of 7 jokes._

## 📝 Command Line Options 📝 (Some of these parameters are not implemented.)

Type --help or use the GUI
```bash
Usage: bark_perform.py [-h] [--text_prompt TEXT_PROMPT] [--list_speakers LIST_SPEAKERS] [--dry_run DRY_RUN] [--text_splits_only TEXT_SPLITS_ONLY] [--history_prompt HISTORY_PROMPT]
                       [--prompt_file PROMPT_FILE] [--split_input_into_separate_prompts_by {word,line,sentence,char,string,random,regex}]
                       [--split_input_into_separate_prompts_by_value SPLIT_INPUT_INTO_SEPARATE_PROMPTS_BY_VALUE] [--always_save_speaker ALWAYS_SAVE_SPEAKER]
                       [--output_iterations OUTPUT_ITERATIONS] [--output_filename OUTPUT_FILENAME] [--output_dir OUTPUT_DIR] [--hoarder_mode HOARDER_MODE] [--extra_stats EXTRA_STATS]
                       [--show_generation_times SHOW_GENERATION_TIMES] [--output_format {wav,mp3,ogg,flac,mp4}] [--text_use_gpu TEXT_USE_GPU] [--text_use_small TEXT_USE_SMALL]
                       [--coarse_use_gpu COARSE_USE_GPU] [--coarse_use_small COARSE_USE_SMALL] [--fine_use_gpu FINE_USE_GPU] [--fine_use_small FINE_USE_SMALL] [--codec_use_gpu CODEC_USE_GPU]
                       [--force_reload FORCE_RELOAD] [--GLOBAL_ENABLE_MPS GLOBAL_ENABLE_MPS] [--USE_SMALL_MODELS USE_SMALL_MODELS] [--OFFLOAD_CPU OFFLOAD_CPU] [--text_temp TEXT_TEMP]
                       [--waveform_temp WAVEFORM_TEMP] [--confused_travolta_mode CONFUSED_TRAVOLTA_MODE] [--silent SILENT] [--seed SEED] [--stable_mode_interval STABLE_MODE_INTERVAL]
                       [--single_starting_seed SINGLE_STARTING_SEED] [--split_character_goal_length SPLIT_CHARACTER_GOAL_LENGTH] [--split_character_max_length SPLIT_CHARACTER_MAX_LENGTH]
                       [--split_character_jitter SPLIT_CHARACTER_JITTER] [--add_silence_between_segments ADD_SILENCE_BETWEEN_SEGMENTS]
                       [--process_text_by_each {word,line,sentence,char,string,random,regex}] [--group_text_by_counting {word,line,sentence,char,string,random,regex}]
                       [--in_groups_of_size IN_GROUPS_OF_SIZE] [--split_type_string SPLIT_TYPE_STRING] [--prompt_text_prefix PROMPT_TEXT_PREFIX]
                       [--extra_confused_travolta_mode EXTRA_CONFUSED_TRAVOLTA_MODE] [--seperate_prompts SEPERATE_PROMPTS] [--semantic_history_only SEMANTIC_HISTORY_ONLY]
                       [--absolute_semantic_history_only ABSOLUTE_SEMANTIC_HISTORY_ONLY] [--absolute_semantic_history_only_every_x ABSOLUTE_SEMANTIC_HISTORY_ONLY_EVERY_X]
                       [--semantic_history_starting_weight SEMANTIC_HISTORY_STARTING_WEIGHT] [--semantic_history_future_weight SEMANTIC_HISTORY_FUTURE_WEIGHT]
                       [--semantic_prev_segment_weight SEMANTIC_PREV_SEGMENT_WEIGHT] [--coarse_history_starting_weight COARSE_HISTORY_STARTING_WEIGHT]
                       [--coarse_history_future_weight COARSE_HISTORY_FUTURE_WEIGHT] [--coarse_prev_segment_weight COARSE_PREV_SEGMENT_WEIGHT]
                       [--fine_history_starting_weight FINE_HISTORY_STARTING_WEIGHT] [--fine_history_future_weight FINE_HISTORY_FUTURE_WEIGHT]
                       [--fine_prev_segment_weight FINE_PREV_SEGMENT_WEIGHT] [--custom_audio_processing_function CUSTOM_AUDIO_PROCESSING_FUNCTION] [--use_smaller_models USE_SMALLER_MODELS]
                       [--bark_cloning_large_model BARK_CLONING_LARGE_MODEL] [--semantic_temp SEMANTIC_TEMP] [--semantic_top_k SEMANTIC_TOP_K] [--semantic_top_p SEMANTIC_TOP_P]
                       [--semantic_min_eos_p SEMANTIC_MIN_EOS_P] [--semantic_max_gen_duration_s SEMANTIC_MAX_GEN_DURATION_S] [--semantic_allow_early_stop SEMANTIC_ALLOW_EARLY_STOP]
                       [--semantic_use_kv_caching SEMANTIC_USE_KV_CACHING] [--semantic_seed SEMANTIC_SEED] [--semantic_history_oversize_limit SEMANTIC_HISTORY_OVERSIZE_LIMIT]
                       [--coarse_temp COARSE_TEMP] [--coarse_top_k COARSE_TOP_K] [--coarse_top_p COARSE_TOP_P] [--coarse_max_coarse_history COARSE_MAX_COARSE_HISTORY]
                       [--coarse_sliding_window_len COARSE_SLIDING_WINDOW_LEN] [--coarse_kv_caching COARSE_KV_CACHING] [--coarse_seed COARSE_SEED]
                       [--x_coarse_history_alignment_hack X_COARSE_HISTORY_ALIGNMENT_HACK] [--fine_temp FINE_TEMP] [--fine_seed FINE_SEED] [--render_npz_samples RENDER_NPZ_SAMPLES]

```

### prompt_file input text file example
```myprompts.txt 
This is the first prompt.
Lots of text here maybe. 
As long as you want.

AAAAA

This is the second prompt.

AAAAA

This is the third prompt.

AAAAA

This is the fourth prompt.
```

```
python bark_perform.py --prompt_file myprompts.txt --split_input_into_separate_prompts_by string --split_input_into_separate_prompts_by_value AAAAA --output_dir myprompts_samples
```


# 🐶 Bark

[![](https://dcbadge.vercel.app/api/server/J2B2vsjKuE?style=flat&compact=True)](https://discord.gg/J2B2vsjKuE)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/OnusFM.svg?style=social&label=@OnusFM)](https://twitter.com/OnusFM)
<a href="http://www.repostatus.org/#active"><img src="http://www.repostatus.org/badges/latest/active.svg" /></a>

[Examples](https://suno-ai.notion.site/Bark-Examples-5edae8b02a604b54a42244ba45ebc2e2) • [Suno Studio Waitlist](https://3os84zs17th.typeform.com/suno-studio) • [Updates](#-updates) • [How to Use](#-usage-in-python) • [Installation](#-installation) • [FAQ](#-faq)

[//]: <br> (vertical spaces around image)
<br>
<p align="center">
<img src="https://user-images.githubusercontent.com/5068315/235310676-a4b3b511-90ec-4edf-8153-7ccf14905d73.png" width="500"></img>
</p>
<br>

Bark is a transformer-based text-to-audio model created by [Suno](https://suno.ai). Bark can generate highly realistic, multilingual speech as well as other audio - including music, background noise and simple sound effects. The model can also produce nonverbal communications like laughing, sighing and crying. To support the research community, we are providing access to pretrained model checkpoints, which are ready for inference and available for commercial use.

## ⚠ Disclaimer
Bark was developed for research purposes. It is not a conventional text-to-speech model but instead a fully generative text-to-audio model, which can deviate in unexpected ways from provided prompts. Suno does not take responsibility for any output generated. Use at your own risk, and please act responsibly.

## 🎧 Demos  

[![Open in Spaces](https://img.shields.io/badge/🤗-Open%20in%20Spaces-blue.svg)](https://huggingface.co/spaces/suno/bark)
[![Open on Replicate](https://img.shields.io/badge/®️-Open%20on%20Replicate-blue.svg)](https://replicate.com/suno-ai/bark)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1eJfA2XUa-mXwdMy7DoYKVYHI1iTd9Vkt?usp=sharing)

## 🚀 Updates

**2023.05.01**
- ©️ Bark is now licensed under the MIT License, meaning it's now available for commercial use!  
- ⚡ 2x speed-up on GPU. 10x speed-up on CPU. We also added an option for a smaller version of Bark, which offers additional speed-up with the trade-off of slightly lower quality. 
- 📕 [Long-form generation](notebooks/long_form_generation.ipynb), voice consistency enhancements and other examples are now documented in a new [notebooks](./notebooks) section.
- 👥 We created a [voice prompt library](https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc67cff786b04b50b3ceb756fd05f68c). We hope this resource helps you find useful prompts for your use cases! You can also join us on [Discord](https://discord.gg/J2B2vsjKuE), where the community actively shares useful prompts in the **#audio-prompts** channel.  
- 💬 Growing community support and access to new features here: 

     [![](https://dcbadge.vercel.app/api/server/J2B2vsjKuE)](https://discord.gg/J2B2vsjKuE)

- 💾 You can now use Bark with GPUs that have low VRAM (<4GB).

**2023.04.20**
- 🐶 Bark release!

## 🐍 Usage in Python

<details open>
  <summary><h3>🪑 Basics</h3></summary>

```python
from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
from IPython.display import Audio

# download and load all models
preload_models()

# generate audio from text
text_prompt = """
     Hello, my name is Suno. And, uh — and I like pizza. [laughs] 
     But I also have other interests such as playing tic tac toe.
"""
audio_array = generate_audio(text_prompt)

# save audio to disk
write_wav("bark_generation.wav", SAMPLE_RATE, audio_array)
  
# play text in notebook
Audio(audio_array, rate=SAMPLE_RATE)
```

[pizza.webm](https://user-images.githubusercontent.com/5068315/230490503-417e688d-5115-4eee-9550-b46a2b465ee3.webm)

</details>

<details open>
  <summary><h3>🌎 Foreign Language</h3></summary>
<br>
Bark supports various languages out-of-the-box and automatically determines language from input text. When prompted with code-switched text, Bark will attempt to employ the native accent for the respective languages. English quality is best for the time being, and we expect other languages to further improve with scaling. 
<br>
<br>

```python

text_prompt = """
    추석은 내가 가장 좋아하는 명절이다. 나는 며칠 동안 휴식을 취하고 친구 및 가족과 시간을 보낼 수 있습니다.
"""
audio_array = generate_audio(text_prompt)
```
[suno_korean.webm](https://user-images.githubusercontent.com/32879321/235313033-dc4477b9-2da0-4b94-9c8b-a8c2d8f5bb5e.webm)
  
*Note: since Bark recognizes languages automatically from input text, it is possible to use for example a german history prompt with english text. This usually leads to english audio with a german accent.*

</details>

<details open>
  <summary><h3>🎶 Music</h3></summary>
Bark can generate all types of audio, and, in principle, doesn't see a difference between speech and music. Sometimes Bark chooses to generate text as music, but you can help it out by adding music notes around your lyrics.
<br>
<br>

```python
text_prompt = """
    ♪ In the jungle, the mighty jungle, the lion barks tonight ♪
"""
audio_array = generate_audio(text_prompt)
```
[lion.webm](https://user-images.githubusercontent.com/5068315/230684766-97f5ea23-ad99-473c-924b-66b6fab24289.webm)
</details>

<details open>
<summary><h3>🎤 Voice Presets</h3></summary>
  
Bark supports 100+ speaker presets across [supported languages](#supported-languages). You can browse the library of speaker presets [here](https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc67cff786b04b50b3ceb756fd05f68c), or in the [code](bark/assets/prompts). The community also often shares presets in [Discord](https://discord.gg/J2B2vsjKuE).

Bark tries to match the tone, pitch, emotion and prosody of a given preset, but does not currently support custom voice cloning. The model also attempts to preserve music, ambient noise, etc.
<br>
<br>

```python
text_prompt = """
    I have a silky smooth voice, and today I will tell you about 
    the exercise regimen of the common sloth.
"""
audio_array = generate_audio(text_prompt, history_prompt="v2/en_speaker_1")
```

[sloth.webm](https://user-images.githubusercontent.com/5068315/230684883-a344c619-a560-4ff5-8b99-b4463a34487b.webm)
</details>

### Generating Longer Audio
  
By default, `generate_audio` works well with around 13 seconds of spoken text. For an example of how to do long-form generation, see this [example notebook](notebooks/long_form_generation.ipynb).

<details>
<summary>Click to toggle example long-form generations (from the example notebook)</summary>

[dialog.webm](https://user-images.githubusercontent.com/2565833/235463539-f57608da-e4cb-4062-8771-148e29512b01.webm)

[longform_advanced.webm](https://user-images.githubusercontent.com/2565833/235463547-1c0d8744-269b-43fe-9630-897ea5731652.webm)

[longform_basic.webm](https://user-images.githubusercontent.com/2565833/235463559-87efe9f8-a2db-4d59-b764-57db83f95270.webm)

</details>




## 💻 Installation

```
pip install git+https://github.com/suno-ai/bark.git
```

or

```
git clone https://github.com/suno-ai/bark
cd bark && pip install . 
```
*Note: Do NOT use 'pip install bark'. It installs a different package, which is not managed by Suno.*


## 🛠️ Hardware and Inference Speed

Bark has been tested and works on both CPU and GPU (`pytorch 2.0+`, CUDA 11.7 and CUDA 12.0).

On enterprise GPUs and PyTorch nightly, Bark can generate audio in roughly real-time. On older GPUs, default colab, or CPU, inference time might be significantly slower. For older GPUs or CPU you might want to consider using smaller models. Details can be found in out tutorial sections here.

The full version of Bark requires around 12GB of VRAM to hold everything on GPU at the same time. 
To use a smaller version of the models, which should fit into 8GB VRAM, set the environment flag `SUNO_USE_SMALL_MODELS=True`.

If you don't have hardware available or if you want to play with bigger versions of our models, you can also sign up for early access to our model playground [here](https://3os84zs17th.typeform.com/suno-studio).

## ⚙️ Details

Bark is fully generative tex-to-audio model devolved for research and demo purposes. It follows a GPT style architecture similar to [AudioLM](https://arxiv.org/abs/2209.03143) and [Vall-E](https://arxiv.org/abs/2301.02111) and a quantized Audio representation from [EnCodec](https://github.com/facebookresearch/encodec). It is not a conventional TTS model, but instead a fully generative text-to-audio model capable of deviating in unexpected ways from any given script. Different to previous approaches, the input text prompt is converted directly to audio without the intermediate use of phonemes. It can therefore generalize to arbitrary instructions beyond speech such as music lyrics, sound effects or other non-speech sounds.

Below is a list of some known non-speech sounds, but we are finding more every day. Please let us know if you find patterns that work particularly well on [Discord](https://discord.gg/J2B2vsjKuE)!

- `[laughter]`
- `[laughs]`
- `[sighs]`
- `[music]`
- `[gasps]`
- `[clears throat]`
- `—` or `...` for hesitations
- `♪` for song lyrics
- CAPITALIZATION for emphasis of a word
- `[MAN]` and `[WOMAN]` to bias Bark toward male and female speakers, respectively

### Supported Languages

| Language | Status |
| --- | --- |
| English (en) | ✅ |
| German (de) | ✅ |
| Spanish (es) | ✅ |
| French (fr) | ✅ |
| Hindi (hi) | ✅ |
| Italian (it) | ✅ |
| Japanese (ja) | ✅ |
| Korean (ko) | ✅ |
| Polish (pl) | ✅ |
| Portuguese (pt) | ✅ |
| Russian (ru) | ✅ |
| Turkish (tr) | ✅ |
| Chinese, simplified (zh) | ✅ |

Requests for future language support [here](https://github.com/suno-ai/bark/discussions/111) or in the **#forums** channel on [Discord](https://discord.com/invite/J2B2vsjKuE). 

## 🙏 Appreciation

- [nanoGPT](https://github.com/karpathy/nanoGPT) for a dead-simple and blazing fast implementation of GPT-style models
- [EnCodec](https://github.com/facebookresearch/encodec) for a state-of-the-art implementation of a fantastic audio codec
- [AudioLM](https://github.com/lucidrains/audiolm-pytorch) for related training and inference code
- [Vall-E](https://arxiv.org/abs/2301.02111), [AudioLM](https://arxiv.org/abs/2209.03143) and many other ground-breaking papers that enabled the development of Bark

## © License

Bark is licensed under the MIT License. 

Please contact us at `bark@suno.ai` to request access to a larger version of the model.  

## 📱 Community

- [Twitter](https://twitter.com/OnusFM)
- [Discord](https://discord.gg/J2B2vsjKuE)

## 🎧 Suno Studio (Early Access)

We’re developing a playground for our models, including Bark. 

If you are interested, you can sign up for early access [here](https://3os84zs17th.typeform.com/suno-studio).

## ❓ FAQ

#### How do I specify where models are downloaded and cached?
* Bark uses Hugging Face to download and store models. You can see find more info [here](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hfhome). 


#### Bark's generations sometimes differ from my prompts. What's happening?
* Bark is a GPT-style model. As such, it may take some creative liberties in its generations, resulting in higher-variance model outputs than traditional text-to-speech approaches.

#### What voices are supported by Bark?  
* Bark supports 100+ speaker presets across [supported languages](#supported-languages). You can browse the library of speaker presets [here](https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc67cff786b04b50b3ceb756fd05f68c). The community also shares presets in [Discord](https://discord.gg/J2B2vsjKuE). Bark also supports generating unique random voices that fit the input text. Bark does not currently support custom voice cloning.

#### Why is the output limited to ~13-14 seconds?
* Bark is a GPT-style model, and its architecture/context window is optimized to output generations with roughly this length.

#### How much VRAM do I need?
* The full version of Bark requires around 12Gb of memory to hold everything on GPU at the same time. However, even smaller cards down to ~2Gb work with some additional settings. Simply add the following code snippet before your generation: 

```python
import os
os.environ["SUNO_OFFLOAD_CPU"] = True
os.environ["SUNO_USE_SMALL_MODELS"] = True
```

#### My generated audio sounds like a 1980s phone call. What's happening?
* Bark generates audio from scratch. It is not meant to create only high-fidelity, studio-quality speech. Rather, outputs could be anything from perfect speech to multiple people arguing at a baseball game recorded with bad microphones.
