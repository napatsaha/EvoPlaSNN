from typing import List
import pandas as pd
import os, yaml, argparse
from pathlib import Path
import numpy as np

# from run_evo_lrule import eval
import snn.plot as snn_plot

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

def make_eval_result(run_dir, num_trials=None, filter_vars=True, change_name=True, save=True, ignore_na=True):
    results_dir = Path(run_dir)   

    if (results_dir / "eval_result.csv").exists():
        existing_evals = pd.read_csv(results_dir / "eval_result.csv", index_col=0, usecols=["run_name", "mean_fts", "std_fts"])
    else:
        existing_evals = pd.DataFrame(columns=["run_name", "mean_fts", "std_fts"])
    existing_evals 

    config_list = []
    num_trials = num_trials

    for dirpath, dirnames, filenames in os.walk(results_dir):
        if "config.yaml" in filenames:
            dirpath = Path(dirpath)
            run_name = dirpath.name
            config_path = os.path.join(dirpath, "config.yaml")
            eval_result_csv = [f for f in filenames if f.endswith("eval_result.csv")]
            if len(eval_result_csv) > 0:
                print(f"Found existing evaluation results for {run_name} at {eval_result_csv[0]}")
                result = pd.read_csv(dirpath / eval_result_csv[0])
                mean_fts = result["fitness"].mean()
                std_fts = result["fitness"].std()
                num_evals = result["trial"].max() + 1
            elif run_name in existing_evals.index:
                print(f"Skipping {run_name} as it has already been evaluated.")
                mean_fts = existing_evals.loc[run_name, "mean_fts"]
                std_fts = existing_evals.loc[run_name, "std_fts"]
                num_evals = num_trials
            else:
                print(f"Skipping {run_name} since it has no evaluation results.")
                # Run evaluation and get mean and std of fitness trials
                # mean_fts, std_fts = eval(dirpath, save_plots=False, num_evals=num_trials, verbose=False)
                # num_evals = num_trials
                mean_fts, std_fts, num_evals = np.nan, np.nan, 0
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                config['run_name'] = run_name
                config['mean_fts'] = np.round(mean_fts, 5)
                config['std_fts'] = np.round(std_fts, 5)
                config['num_evals'] = num_evals
                config_list.append(config)

    # Convert list of dicts to DataFrame
    cfgs = pd.json_normalize(config_list, sep="__").set_index('run_name', drop=True)
    # Select only variables with differing values (disregarding NA)
    if filter_vars:
        for col in cfgs.columns:
            if cfgs[col].dtype == "O" and isinstance(cfgs[col].iloc[0], list):
                cfgs[col] = cfgs[col].astype("str")
        main_vars = cfgs.nunique(axis=0, dropna=ignore_na) > 1
        main_vars[["num_sim_steps", "num_evals"]] = True
        cfgs = cfgs.loc[:, main_vars]
    # Removes "aaa_params__" in config names
    if change_name:
        cfgs.rename(columns=convert_name, inplace=True)
    # Conbime rewarder_type and fitness_type into one column
    # cfgs["fitness_type"] = cfgs["fitness_type"].where(cfgs["rewarder_type"] != "weighted", cfgs["rewarder_type"])
    # cfgs.drop(columns=["rewarder_type"], inplace=True)
    # For this experiment, before clip_weights was specified in config, it has a default of True
    # cfgs.fillna({"clip_weights": True}, inplace=True)
    # Sort by chronological order
    cfgs.sort_index(inplace=True)
    # cfgs = cfgs.round({"target_fitness": 1})
    if save:
        cfgs.to_csv(results_dir / "eval_result.csv")
    return cfgs


def collate_raw_results(exp_path: List | Path | str):
    subdir_filename = "eval_rule_01_eval_result.csv"
    subdata_rec = []
    if isinstance(exp_path, str):
        exp_path = [Path(exp_path)]
    elif isinstance(exp_path, Path):
        exp_path = [exp_path]
    elif isinstance(exp_path, list):
        exp_path = [Path(p) if isinstance(p, str) else p for p in exp_path]
    for sub_dir in exp_path:
        for dirpath, dirnames, filenames in sub_dir.walk():
            if subdir_filename in filenames:
                subdata = pd.read_csv(Path(dirpath, subdir_filename))
                subdata["run_name"] = dirpath.name
                subdata_rec.append(subdata)
    subdata_df = pd.concat(subdata_rec, ignore_index=True, axis=0)
    return subdata_df


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("run_dir", type=str, default=None,
                        help="Directory name where the results are stored.")
    argparser.add_argument("--num_trials", type=int, default=100,
                        help="Number of trials to run for each evaluation.")
    argparser.add_argument("--plot", "-p", action="store_true", default=False,
                        help="Whether or not to plot comparison between different runs in current experiment dir." \
                        "(Requires x_var, hue_var, and col_var (optional) to be set.)")
    argparser.add_argument("--x_var", type=str, default=None,
                        help="Variable to use for x-axis in evaluation comparison plots.")
    argparser.add_argument("--hue_var", type=str, default=None,
                        help="Variable to use for hue in evaluation comparison plots.")
    argparser.add_argument("--col_var", type=str, default=None,
                        help="Variable to use for columns in evaluation comparison plots.")
    args = argparser.parse_args()

    make_eval_result(run_dir=args.run_dir, num_trials=args.num_trials)
    if args.plot:
        if args.x_var is None or args.hue_var is None:
            raise ValueError("x_var and hue_var must be specified for plotting comparison results.")
        snn_plot.plot_compare_run(args.run_dir, x_var=args.x_var, hue_var=args.hue_var, col_var=args.col_var,
                              savepath=f"{args.run_dir}/eval_comparison.png", show=False)