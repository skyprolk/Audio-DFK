import argparse
import glob
import os
import shutil
import site
import subprocess
import sys

script_dir = os.getcwd()
conda_env_path = os.path.join(script_dir, "installer_files", "env")

print(f"script_dir: {script_dir}")

print(f"conda_env_path: {conda_env_path}")

conda_environment_yml = os.path.join(script_dir, "bark")
conda_environment_yml_file = f"{conda_environment_yml}\environment-cuda.yml"
# CMD_FLAGS = '--share'

CMD_FLAGS = ''

"""
# Gradio flags

--share               Enable share setting.
--user USER           User for authentication.
--password PASSWORD   Password for authentication.
--listen              Server name setting.
--server_port SERVER_PORT
                    Port setting.
--no-autolaunch       Disable automatic opening of the app in browser.
--debug               Enable detailed error messages and extra outputs.
--incolab             Default for Colab.
"""

# Allows users to set flags in "OOBABOOGA_FLAGS" environment variable
if "OOBABOOGA_FLAGS" in os.environ:
    CMD_FLAGS = os.environ["OOBABOOGA_FLAGS"]
    # print("The following flags have been taken from the environment variable 'OOBABOOGA_FLAGS':")
    # print(CMD_FLAGS)
    # print("To use the CMD_FLAGS Inside webui.py, unset 'OOBABOOGA_FLAGS'.\n")


def print_big_message(message):
    message = message.strip()
    lines = message.split('\n')
    print("\n\n*******************************************************************")
    for line in lines:
        if line.strip() != '':
            print("*", line)

    print("*******************************************************************\n\n")


def run_cmd(cmd, assert_success=False, environment=False, capture_output=False, env=None):
    # Use the conda environment
    if environment:
        if sys.platform.startswith("win"):
            conda_bat_path = os.path.join(script_dir, "installer_files", "conda", "condabin", "conda.bat")
            cmd = "\"" + conda_bat_path + "\" activate \"" + conda_env_path + "\" >nul && " + cmd
        else:
            conda_sh_path = os.path.join(script_dir, "installer_files", "conda", "etc", "profile.d", "conda.sh")
            cmd = ". \"" + conda_sh_path + "\" && conda activate \"" + conda_env_path + "\" && " + cmd

    # Run shell commands
    print(f"Running command: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=capture_output, env=env)

    # Assert the command ran successfully
    if assert_success and result.returncode != 0:
        print("Command '" + cmd + "' failed with exit status code '" + str(result.returncode) + "'. Exiting...")
        sys.exit()

    return result


def check_env():
    # If we have access to conda, we are probably in an environment
    conda_exist = run_cmd("conda", environment=True, capture_output=True).returncode == 0
    if not conda_exist:
        print("Conda is not installed. Exiting...")
        sys.exit()

    # Ensure this is a new environment and not the base environment
    if os.environ["CONDA_DEFAULT_ENV"] == "base":
        print("Create an environment for this project and activate it. Exiting...")
        sys.exit()


def install_dependencies():
    # Select your GPU or, choose to run in CPU mode

    """
    print("What is your GPU")
    print()
    print("A) NVIDIA")
    print("B) AMD")
    print("C) Apple M Series")
    print("D) None (I want to run in CPU mode)")
    print()
    gpuchoice = input("Input> ").lower()
    """
    gpuchoice = "a"


    if gpuchoice == "d":
        print_big_message("This installer is only for NVIDIA at the moment. Just testing.")

    if os.path.exists("bark/"): 

        run_cmd(f"conda env update -p \"{conda_env_path}\" -f {conda_environment_yml_file}  --prune --solver=libmamba", assert_success=True, environment=True)
    else:
        run_cmd("conda install -n base -y conda-libmamba-solver", assert_success=True, environment=True)
        run_cmd("conda install -y -k git --solver=libmamba", assert_success=True, environment=True)
        run_cmd("conda install -y -k pip --solver=libmamba", assert_success=True, environment=True)
        run_cmd("pip install fairseq@https://github.com/Sharrnah/fairseq/releases/download/v0.12.4/fairseq-0.12.4-cp310-cp310-win_amd64.whl", assert_success=False, environment=True)

        run_cmd("git clone https://github.com/JonathanFly/bark.git", assert_success=True, environment=True)
    
        run_cmd(f"conda env update -p \"{conda_env_path}\" -f {conda_environment_yml_file} --prune --solver=libmamba", assert_success=True, environment=True)

        # this fights with conda-forge cuda-toolkit and keeps flipping the lirary versions
        # run_cmd("conda install -y -c \"nvidia/label/cuda-11.8.0\" cuda-toolkit --solver=libmamba", assert_success=False, environment=True) # later for more performance

        run_cmd("ffdl install --add-path", assert_success=False, environment=True)

    print(f"\n\nIf this is your first time installing Bark, after this installation is done, close this window.\n\nClose any text terminals open.\n\nThen click on start_bark_infinity.bat in a fresh explorer window. This seems to be necessary for FFMPEG to be detected after installation. Then click on \"start_up_already_installed_bark_infinity_windows.bat\"")
    return
    


def launch_webui():
    os.chdir("bark")
    run_cmd(f"python bark_webui.py {CMD_FLAGS}", environment=True)


if __name__ == "__main__":
    # Verifies we are in a conda environment
    check_env()

    parser = argparse.ArgumentParser()
    parser.add_argument('--update', action='store_true', help='Update the web UI.')
    args = parser.parse_args()

    if args.update:
        pass 
        # update_dependencies()
    else:
        # If webui has already been installed, skip and run
        if not os.path.exists("bark/") or True:
            install_dependencies()
            os.chdir(script_dir)

