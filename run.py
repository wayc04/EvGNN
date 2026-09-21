import copy
import os
import shlex

from configs import MAIN_EXPERIMENT_PRESETS


DEFAULT_OPTIONS = {
    "epochs": 1000,
}


def generate_command(options):
    cmd = ["python3", "train.py"]
    for key, value in options.items():
        cmd.extend([f"--{key}", str(value)])
    return " ".join(shlex.quote(part) for part in cmd)


def run(options):
    os.system(generate_command(copy.deepcopy(options)))


def run_preset(name, extra_options=None):
    options = dict(DEFAULT_OPTIONS)
    options["preset"] = name
    if extra_options:
        options.update(extra_options)
    run(options)


if __name__ == "__main__":
    for preset in MAIN_EXPERIMENT_PRESETS:
        run_preset(preset)
