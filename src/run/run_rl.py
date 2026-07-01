import time
import argparse
import yaml
from pathlib import Path

from common.utils import parse_config_overrides, update_dictionary
from run.evaluate import evaluate_and_plot
from common.utils import create_solver
from evo.manager import EvoManager


ROOT = Path(__file__).parent.parent.parent


def main(config_file: str | Path | dict, *, config_overrides: dict = None, parent_run: str = None, default_dir: str = "binary_es") -> Path:
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

    # Directory to save new results

    
    if parent_run is not None:
        results_path = Path(ROOT, parent_run)
    else:
        results_path = Path(ROOT, "results", default_dir)
    results_path = results_path / time.strftime("%y-%m-%d_%H-%M-%S")
    results_path.mkdir(parents=True, exist_ok=True)

    # manager_type = config["evo_params"]["manager"].pop("type", "original")
    # Rename config["arule_params"] to config["lrule_params"] if exist
    # if "arule_params" in config:
    #     config["lrule_params"] = config.get("lrule_params", {}).update(config["arule_params"])
    #     del config["arule_params"]

    # # Configure SNN Evaluator object
    # evaluator: Evaluator = RL_Evaluator(
    #     params=config,
    #     record_info=False,
    #     **config["evo_params"]["evaluator"]
    # )

    # # Configure Evolution Solver object
    # # if config["lrule_params"]["type"] == "ann":
    # #     config["evo_params"]["solver"]["ndim"] = evaluator.get_parameter_size()
    # # TODO: Calculate this without Evaluator
    # config["evo_params"]["solver"]["minimise"] = evaluator.is_minimise()
    # solver = create_solver(config["evo_params"]["solver"], genome_params=config.get("lrule_params").copy())
    # if "popsize" not in config["evo_params"]["solver"]:
    #     config["evo_params"]["solver"]["popsize"] = solver.popsize
    # if "ndim" not in config["evo_params"]["solver"]:
    #     config["evo_params"]["solver"]["ndim"] = solver.ndim
    
    # # use_target_fitness = config["evo_params"]["manager"].get("use_target_fitness", False)
    # # if use_target_fitness:
    # # TODO: Calculate this without Evaluator
    # config["evo_params"]["manager"]["target_fitness"] = evaluator.get_target_fitness()
    
    # Configure Evolution Manager object
    manager = EvoManager(config, results_path=results_path, **config["evo_params"]["manager"])
    config = manager.config

    # Re-insert the manager type into the config
    # config["evo_params"]["manager"]["type"] = manager_type

    # Save a copy of configuration used
    with open(results_path / "config.yaml", "w") as f:
        yaml.dump(config, f, sort_keys=False)

    # Begin experiment
    manager.run()

    print(f"Evolution run completed. Results saved to {results_path}")

    return results_path 


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