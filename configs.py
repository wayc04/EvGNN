"""Experiment presets used by the EvGNN training scripts."""

PRESET_FIELDS = (
    "dataset",
    "noise",
    "seed",
    "label_rate",
    "weight_all",
    "weight_con",
    "weight_pse",
    "lr",
    "hidden1",
    "hidden",
)

EXPERIMENT_CONFIGS = {
    "cs_uniform": {
        "dataset": "cs",
        "noise": "uniform",
        "seed": 0,
        "label_rate": 0.01,
        "weight_all": 0.01,
        "weight_con": 0.001,
        "weight_pse": 1,
        "lr": 0.0001,
        "hidden1": 256,
        "hidden": 32,
    },
    "photo_uniform": {
        "dataset": "photo",
        "noise": "uniform",
        "seed": 15,
        "label_rate": 0.01,
        "weight_all": 0.1,
        "weight_con": 1,
        "weight_pse": 0.01,
        "lr": 0.001,
        "hidden1": 256,
        "hidden": 256,
    },
    "computers_uniform": {
        "dataset": "computers",
        "noise": "uniform",
        "seed": 121,
        "label_rate": 0.01,
        "weight_all": 1,
        "weight_con": 0.001,
        "weight_pse": 1,
        "lr": 0.001,
        "hidden1": 256,
        "hidden": 128,
    },
    "pubmed_uniform": {
        "dataset": "pubmed",
        "noise": "uniform",
        "seed": 0,
        "label_rate": 0.01,
        "weight_all": 0.1,
        "weight_con": 0.001,
        "weight_pse": 1,
        "lr": 0.0001,
        "hidden1": 16,
        "hidden": 8,
    },
    "dblp_uniform": {
        "dataset": "dblp",
        "noise": "uniform",
        "seed": 0,
        "label_rate": 0.01,
        "weight_all": 0.01,
        "weight_con": 10,
        "weight_pse": 10,
        "lr": 0.001,
        "hidden1": 2048,
        "hidden": 32,
    },
    "citeseer_uniform": {
        "dataset": "citeseer",
        "noise": "uniform",
        "seed": 121,
        "label_rate": 0.05,
        "weight_all": 0.1,
        "weight_con": 0.001,
        "weight_pse": 50,
        "lr": 0.001,
        "hidden1": 256,
        "hidden": 256,
    },
    "cs_pair": {
        "dataset": "cs",
        "noise": "pair",
        "seed": 0,
        "label_rate": 0.01,
        "weight_all": 0.01,
        "weight_con": 0.001,
        "weight_pse": 1,
        "lr": 0.0001,
        "hidden1": 256,
        "hidden": 32,
    },
    "photo_pair": {
        "dataset": "photo",
        "noise": "pair",
        "seed": 0,
        "label_rate": 0.01,
        "weight_all": 0.1,
        "weight_con": 10,
        "weight_pse": 1,
        "lr": 0.001,
        "hidden1": 256,
        "hidden": 256,
    },
    "computers_pair": {
        "dataset": "computers",
        "noise": "pair",
        "seed": 15,
        "label_rate": 0.01,
        "weight_all": 1,
        "weight_con": 0.001,
        "weight_pse": 1,
        "lr": 0.001,
        "hidden1": 256,
        "hidden": 128,
    },
    "pubmed_pair": {
        "dataset": "pubmed",
        "noise": "pair",
        "seed": 50,
        "label_rate": 0.01,
        "weight_all": 0.1,
        "weight_con": 0.001,
        "weight_pse": 1,
        "lr": 0.0001,
        "hidden1": 16,
        "hidden": 8,
    },
    "dblp_pair": {
        "dataset": "dblp",
        "noise": "pair",
        "seed": 20,
        "label_rate": 0.01,
        "weight_all": 0.01,
        "weight_con": 10,
        "weight_pse": 10,
        "lr": 0.001,
        "hidden1": 2048,
        "hidden": 32,
    },
    "citeseer_pair": {
        "dataset": "citeseer",
        "noise": "pair",
        "seed": 0,
        "label_rate": 0.05,
        "weight_all": 0.1,
        "weight_con": 0.001,
        "weight_pse": 50,
        "lr": 0.001,
        "hidden1": 256,
        "hidden": 256,
    },
}

MAIN_EXPERIMENT_PRESETS = tuple(EXPERIMENT_CONFIGS)


def explicit_cli_options(argv):
    supplied = set()
    for item in argv[1:]:
        if item.startswith("--"):
            supplied.add(item[2:].split("=", 1)[0].replace("-", "_"))
    return supplied


def apply_preset(args, explicit_options=None):
    if args.preset is None:
        return args

    explicit_options = explicit_options or set()
    for key, value in EXPERIMENT_CONFIGS[args.preset].items():
        if key not in explicit_options:
            setattr(args, key, value)
    return args
