"""
2026-06-30

Experiment which uses Evaluation from Custom Code (this project) but Evolutionary Algorithm from pyribs package.

Relies on Emitter and Archive only (no Scheduler)
"""
import time
import argparse
import yaml
import logging
import csv
import copy
from pathlib import Path
from tqdm import tqdm
import numpy as np

from common.utils import parse_config_overrides, update_dictionary
from common.base import Evaluator
from run.evaluate import evaluate_and_plot
from evo.manager import EvoManager
from rl.eval import RL_Evaluator

import ribs.archives, ribs.emitters
# from ribs.archives import ProximityArchive
# from ribs.emitters import EvolutionStrategyEmitter
# from ribs.emitters.rankers import NSLCRanker


ROOT = Path(__file__).parent.parent.parent


class Solution_Writer:
    def __init__(self):
        pass

    def setup_logger(self, log_path, field_names):
        self.log_path = Path(log_path)
        self.field_names = field_names
        self._solution_file = open(self.log_path / "solutions.csv", 'w')
        self._writer = csv.DictWriter(self._solution_file, fieldnames=field_names)
        self._writer.writeheader()

    def write_solution(self, gen_no: int, num_indivs: int, **kwargs):
        """
        Write solution in current generation to file
        """
        for i in range(num_indivs):
            row = dict(
                gen=int(gen_no) - 1,
                indiv=i
            )
            for field in self.field_names:
                if field in ('gen', 'indiv'):
                    continue
                data = kwargs.get(field, None)
                if data is None:
                    continue
                row_data = data[i]
                if isinstance(row_data, np.ndarray) and len(row_data) > 1:
                    row_data = list(row_data)
                row[field] = row_data
            self._writer.writerow(row)

    def close(self):
        self._solution_file.close()

    def save_best(self, solutions, fitnesses, save_dir: str | Path, n: int = 1, precision: int = 6):
        top_indices = np.argsort(fitnesses) # Will arrange from lowest to highest fitness
        top_indices = top_indices[:n]
        top_solutions = np.take(solutions, top_indices, axis=0)
        for i, sol in enumerate(top_solutions):
            np.savetxt(Path(save_dir) / f"best_rule_{i+1:02d}.txt", sol, fmt=f'%.{precision}f')


def main(config_file: str | Path | dict, *, config_overrides: dict = None, parent_run: str = None, default_dir: str = "pyribs") -> Path:
    """Main function to run the evolutionary learning rule experiment."""
    # Config file handling
    if isinstance(config_file, dict):
        config = config_file
    elif isinstance(config_file, str | Path):
        if Path(config_file).exists():
            config_path = Path(config_file)
        else:
            config_path = Path(ROOT, "config", config_file)
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        raise ValueError("config_file must be a path to a YAML file or a dictionary.")

    # Apply overrides to the configuration
    if config_overrides:
        config = update_dictionary(config, config_overrides)
    out_config = copy.deepcopy(config)

    # Directory to save new results
    if parent_run is not None:
        results_path = Path(ROOT, parent_run)
    else:
        results_path = Path(ROOT, "results", default_dir)
    results_path = results_path / time.strftime("%y-%m-%d_%H-%M-%S")
    results_path.mkdir(parents=True, exist_ok=True)

    num_gens = config["evo_params"].get("num_gens")
    num_trials = config["evo_params"].get("num_trials")

    ## Begin object initialisation
    # Evaluator
    evaluator: Evaluator = RL_Evaluator(
                    params=config,
                    record_info=False,
                    **config["evaluator_params"]
                )
    
    bounds = [(g.low, g.high) for g in evaluator.dummy_rule.genome.genes for _ in range(g.length)]
    solution_dim = evaluator.get_parameter_size()
    behaviour_dim = evaluator.bc_dim

    # Archive
    arc_cls_name = config["evo_params"]["archive_params"].pop("type")
    arc_cls_obj = getattr(ribs.archives, arc_cls_name)
    archive: ribs.archives.ProximityArchive = arc_cls_obj(
        solution_dim=solution_dim,
        measure_dim=behaviour_dim,
        **config["evo_params"]["archive_params"]
    )
    config["evo_params"]["archive_params"]["solution_dim"] = archive.solution_dim
    config["evo_params"]["archive_params"]["measure_dim"] = archive.measure_dim

    # Emitter init
    emt_cls_name = config["evo_params"]["emitter_params"].pop("type")
    emt_cls_obj = getattr(ribs.emitters, emt_cls_name)
    emitter: ribs.emitters.EvolutionStrategyEmitter = emt_cls_obj(
        archive,
        x0=evaluator.dummy_rule.parameters,
        bounds=bounds,
        **config["evo_params"]["emitter_params"]
    )
    config["evo_params"]["emitter_params"]["bounds"] = bounds

    # Save a copy of configuration used
    with open(results_path / "config.yaml", "w") as f:
        yaml.dump(out_config, f, sort_keys=False)

    # Prepare loggers
    gen_logger = _setup_logger(results_path / "log_generation.log", "Manager")
    evaluator.setup_logger(results_path)
    solution_writer = Solution_Writer()
    solution_writer.setup_logger(results_path, field_names=[
            "gen", "indiv", "status", "local_competition", "novelty", "objective", "genome", "behaviour"
        ])

    # Begin evolution run
    t0 = time.time()
    gen_logger.info("Starting evolutionary optimisation.")
    for gen_count in tqdm(range(num_gens), total=num_gens, desc="Generations", position=0, leave=True):
        # Generate population in current generation
        solutions = emitter.ask()

        # Evaluate each solution one by one
        fitnesses, measures = [], []
        for inv, sol in tqdm(enumerate(solutions), desc="Populations", total=emitter.batch_size, position=1, leave=False):
            fts_list, avg_fts, std_fts, behv = evaluator.evaluate(sol, num_trials=num_trials, gen_count=gen_count, inv_count=inv)
            # Since pyribs algorithms assume minimisation problem, we ensure fitness measure is reversed if maximised
            if not evaluator.is_minimise():
                avg_fts = -avg_fts
            fitnesses.append(avg_fts)
            measures.append(behv)
        measures = np.r_[measures]

        # Update Emitter and Archive with evaluation information
        add_info = archive.add(solutions, fitnesses, measures)
        emitter.tell(solutions, fitnesses, measures, add_info)

        # Record batch info in this generation
        solution_writer.write_solution(gen_no=gen_count, num_indivs=len(solutions),
                                       status=add_info.get("status"),
                                       local_competition=add_info.get("local_competition"),
                                       novelty=add_info.get("novelty"),
                                       objective=fitnesses,
                                       genome=solutions,
                                       behaviour=measures,
                                       )

        # Log result
        best_fitness = min(fitnesses)
        global_best_fitness = archive.best_elite.get('objective')
        gen_logger.info(f"Generation {gen_count}: Best fitness this generation = {best_fitness:.3f}, All-time best fitness = {global_best_fitness:.3f}")
        tqdm.write(f"Generation: {gen_count}. Archive size: {archive.stats.num_elites}. Best fitness: {best_fitness:.3f}")

    t1 = time.time()
    dt = t1 - t0

    ## Wrapping up
    # Saving genome of best n individuals from Archive (Different from previously)
    if archive.stats.num_elites > config["evo_params"]["save_best"]:
        solution_writer.save_best(archive.data('solution'), archive.data('objective'), n=config["evo_params"]["save_best"],
                                save_dir=results_path)
    else:
        tqdm.write("Cannot save elites since number of elites: %d is less than requested 'save_best' %d" % \
                    (archive.stats.num_elites, config["evo_params"]["save_best"]))

    # Log closing info
    best_solution = archive.best_elite
    gen_logger.info("Terminating Evolutionary optimisation.")
    gen_logger.info(f"Best solution: {best_solution.get('solution')}")
    gen_logger.info(f"Best fitness: {best_solution.get('objective'):.3f}")
    gen_logger.info(f"Total time taken: {dt // 3600} hours, {(dt % 3600) // 60} minutes, {dt % 60:.2f} seconds")
    gen_logger.info(f"Total generations: {gen_count}")
    gen_logger.info(f"Results saved to directory: {results_path}")

    print(f"Evolution run completed. Results saved to {results_path}")

    return results_path


def _setup_logger(log_path, logger_name):
    """
    Set up channels for outputting logging information.
    By default, prints to console.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler() if log_path is None else logging.FileHandler(log_path)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    if logger.hasHandlers():
        # Prevent adding multiple handlers if logger is already configured
        logger.handlers.clear()
    logger.addHandler(handler)
    return logger


if __name__ == "__main__":
    # Argument parser
    argparser = argparse.ArgumentParser(description="Run Evolutionary Learning Rule Experiment")
    argparser.add_argument("--config", "-c", type=str, default="binary_es_v2.yaml", help="Path to the configuration file")
    argparser.add_argument("--override", "-o", type=str, nargs="*", help="Override specific config values (e.g., snn_params.neuron_params.tau_mem=0.05)")
    argparser.add_argument("--parent", "-p", type=str, default=None, help="Parent run directory to save results in. Useful for running series of related results.")
    argparser.add_argument("--num_evals", "-e", type=int, default=10, help="Number of evaluation trials to run for the best rule at the end.")
    argparser.add_argument("--plot", "-t", action=argparse.BooleanOptionalAction, default=True, type=bool, help="Specify --no-plot to avoid plot at the end (default is to plot)")
    args = argparser.parse_args()

    # Parse overrides
    config_overrides = parse_config_overrides(args.override) if args.override else {}

    # Run main function
    results_path = main(config_file=args.config, config_overrides=config_overrides, parent_run=args.parent)
    # Evaluation of best solution
    evaluate_and_plot(results_path, save_plots=args.plot, save_results=True, num_evals=args.num_evals, verbose=True)