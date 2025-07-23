import pandas as pd
import os, yaml, argparse
from pathlib import Path

from run_evo_lrule import eval

def convert_name(x):
    tokens = x.split('__')
    if tokens[-1] == "type":
        if tokens[-2].endswith("_params"):
            prefix = tokens[-2].split("_")[0]
        else:
            prefix = tokens[-2]
        return prefix + "_" + tokens[-1]
    else:
        return tokens[-1]

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("group_dir", type=str, default=None,
                        help="Directory name where the results are stored.")
    argparser.add_argument("--num_trials", type=int, default=100,
                        help="Number of trials to run for each evaluation.")
    args = argparser.parse_args()

    results_dir = Path(f"results/binary_es/{args.group_dir}")   

    if (results_dir / "eval_result.csv").exists():
        existing_evals = pd.read_csv(results_dir / "eval_result.csv", index_col=0, usecols=["run_name", "mean_fts", "std_fts"])
    else:
        existing_evals = pd.DataFrame(columns=["run_name", "mean_fts", "std_fts"])
    existing_evals 

    config_list = []
    num_trials = args.num_trials

    for dirpath, dirnames, filenames in os.walk(results_dir):
        if "config.yaml" in filenames:
            run_name = Path(dirpath).name
            config_path = os.path.join(dirpath, "config.yaml")
            if run_name in existing_evals.index:
                print(f"Skipping {run_name} as it has already been evaluated.")
                mean_fts = existing_evals.loc[run_name, "mean_fts"]
                std_fts = existing_evals.loc[run_name, "std_fts"]
            else:
                print(f"Evaluating {run_name}...")
                # Run evaluation and get mean and std of fitness trials
                mean_fts, std_fts = eval(Path(dirpath), save_plots=False, num_evals=num_trials, verbose=False)
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                config['run_name'] = run_name
                config['mean_fts'] = mean_fts.round(5)
                config['std_fts'] = std_fts.round(5)
                config_list.append(config)

    # Convert list of dicts to DataFrame
    cfgs = pd.json_normalize(config_list, sep="__").set_index('run_name', drop=True)
    # Select only variables with differing values (disregarding NA)
    main_vars = cfgs.nunique(axis=0, dropna=True) > 1
    cfgs = cfgs.loc[:, main_vars]
    # Removes "aaa_params__" in config names
    cfgs.rename(columns=convert_name, inplace=True)
    # Conbime rewarder_type and fitness_type into one column
    cfgs["fitness_type"] = cfgs["fitness_type"].where(cfgs["rewarder_type"] != "weighted", cfgs["rewarder_type"])
    cfgs.drop(columns=["rewarder_type"], inplace=True)
    # For this experiment, before clip_weights was specified in config, it has a default of True
    cfgs.fillna({"clip_weights": True}, inplace=True)
    # Sort by chronological order
    cfgs.sort_index(inplace=True)
    cfgs = cfgs.round({"target_fitness": 1})

    cfgs.to_csv(results_dir / "eval_result.csv")