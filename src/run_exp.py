"""
2025-07-31
Takes in a dictionary of config options and perform Randomised search on each non-scalar values in config
"""

import numpy as np
import pandas as pd
import scipy
from scipy import stats
import run_evo_lrule, eval_group
# from .run_evo_lrule import main, eval
from pathlib import Path
import yaml
import shutil
import argparse
import logging
import time


ROOT = Path(__file__).parent.parent


def sample_dict(d: dict):
    new_dict = {}
    for key, val in d.items():
        if isinstance(val, list):
            new_dict[key] = np.random.choice(val).item()
        elif isinstance(val, str) and val.endswith(")"):
            new_dict[key] = eval(val).rvs(1).round(4).item()
        elif isinstance(val, dict):
            new_dict[key] = sample_dict(val)
        else:
            new_dict[key] = val
    return new_dict

def run(search_config: str | Path, exp_dir: str | Path, num_search: int, num_rep: int, num_evals: int):
    if Path(search_config).exists():
        search_config = Path(search_config)
    elif Path(ROOT, "config", search_config).exists():
        search_config = Path("config", search_config)
    else:
        raise FileNotFoundError(f"Search config {search_config} not found.")
    
    # Set up experiment directory
    exp_dir = Path(ROOT, "results", exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    if not exp_dir.is_dir():
        exp_dir = exp_dir.parent
    shutil.copy2(search_config, exp_dir / "search_config.yaml")

    # Set up logging
    logger = logging.getLogger("run_exp")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(exp_dir / "run_exp.log")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Load search space configuration
    with open(search_config, 'r') as f:
        search_dict = yaml.safe_load(f)

    # Prepare eval_result data frame
    result_file = exp_dir / "eval_result.csv"
    if not result_file.exists():
        params_names_flat = pd.json_normalize(search_dict, sep="__").columns
        search_df = pd.DataFrame(columns=["run_name", "mean_fts", "std_fts", "num_evals"] + params_names_flat.to_list())
        search_df.to_csv(result_file, index=False, mode='w', header=True)

    # Sample from search space with each new iterations for num_search times
    ts = []
    t0 = time.time()
    for si in range(num_search):
        new_config = sample_dict(search_dict)
        new_df = pd.json_normalize(new_config, sep="__")
        old_columns = new_df.columns
        # Repeat same config for num_rep times
        for ri in range(num_rep):
            t_rep0 = time.time()
            try:
                logger.info(f"Running iteration {si+1:02}/{num_search}, repetition {ri+1}/{num_rep}:")
                run_path = run_evo_lrule.main(new_config, parent_run=exp_dir)
                mean_fts, std_fts = run_evo_lrule.eval(run_path, save_plots=True, save_results=True, num_evals=num_evals, return_evaluator=False)
                # Save evaluation result (+full config used) straight away to shared file
                run_name = run_path.name
                mean_fts = np.round(mean_fts, 5)
                std_fts = np.round(std_fts, 5)

            except Exception as e:
                logger.info(f"Exception occurred: {e}")
                logger.info(f"Error running on iteration {si}, repeat {ri}. Skipping this configuration.")
                run_name = f"error_{si}_{ri}"
                mean_fts = np.nan
                std_fts = np.nan
                
            t_rep1 = time.time()
            logger.info(f"Run Duration: {t_rep1 - t_rep0:.2f} seconds.")
            ts.append(t_rep1 - t_rep0)

            new_df["run_name"] = run_name
            new_df["mean_fts"] = mean_fts
            new_df["std_fts"] = mean_fts
            new_df["num_evals"] = num_evals
            new_df = new_df[["run_name", "mean_fts", "std_fts", "num_evals"] + old_columns.to_list()]
            new_df.to_csv(result_file, index=False, mode='a', header=False)
    t1 = time.time()
    delta = t1 - t0
    logger.info(f"Total time taken: {delta // (24*60*60)} days, {delta // 3600} hours, {delta // 60} minutes, {delta % 60:.2f} seconds.")
    logger.info(f"Total iterations: {num_search * num_rep}.")
    avg_dur = np.mean(ts)
    logger.info(f"Average time per iteration: {avg_dur // 60:.2f} minutes, {avg_dur % 60:.2f} seconds.")
    max_dur = np.max(ts)
    logger.info(f"Longest iteration: {max_dur // 60:.2f} minutes, {max_dur % 60:.2f} seconds.")
    logger.info(f"Results saved to {result_file}")
    # eval_group.make_eval_result(exp_dir, save_all=True)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("search_file", type=str, help="Path to the file which contains a nested dictionary of search space " \
    "where the values are either static, list of options, or scipy.stats distribution.")
    argparser.add_argument("--exp_dir", "-d", type=str, help="Directory to save the results of the search.")
    argparser.add_argument("--num_search", "-n", type=int, default=20, help="Number of config samples to draw from the search space (assume Randomised Search).")
    argparser.add_argument("--num_rep", "-r", type=int, default=1, help="Number of repeated evolution run to perform for each configuration sampled.")
    argparser.add_argument("--num_evals", "-e", type=int, default=100, help="Number of evaluation trials to run at the end of each evolution run.")
    args = argparser.parse_args()

    run(args.search_file, args.exp_dir, args.num_search, args.num_rep, args.num_evals)
