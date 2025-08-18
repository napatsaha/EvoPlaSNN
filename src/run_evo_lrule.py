import time
import argparse
from typing import Tuple
from lrule.ann import read_ANN_Rule
from lrule.base import LearningRule
import yaml
from pathlib import Path
import numpy as np

from snn.eval import SNN_Evaluator
# from evo.es import EvolutionStrategy
from evo.utils import create_solver

# from snn.plot import plot_weight_over_time, plot_weights, plot_spikes, plot_membranes
import snn.plot as snn_plot


ROOT = Path(__file__).parent.parent


def parse_config_overrides(overrides: list[str]) -> dict:
    """Parse key-value pairs for configuration overrides."""
    config_updates = {}
    for override in overrides:
        keys, value = override.split("=")
        keys = keys.split(".")
        current = config_updates
        for key in keys[:-1]:
            current = current.setdefault(key, {})
        current[keys[-1]] = yaml.safe_load(value)  # Convert value to appropriate type
    return config_updates


def update_dictionary(config: dict, overrides: dict) -> dict:
    """Update a configuration dictionary with overrides."""
    for key in overrides.keys():
        if key in config:
            if isinstance(config[key], dict) and isinstance(overrides[key], dict):
                config[key] = update_dictionary(config[key], overrides[key])
            else:
                config[key] = overrides[key]
        else:
            config[key] = overrides[key]

    return config


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

    manager_type = config["evo_params"]["manager"].pop("type", "original")

    if manager_type == "original":
        from evo.manager import EvoManager
        # Configure SNN Evaluator object
        evaluator = SNN_Evaluator(
            params=config,
            # **config["evo_params"]["evaluator"]
        )

        # Configure Evolution Solver object
        config["evo_params"]["solver"]["ndim"] = evaluator.get_parameter_size()
        config["evo_params"]["solver"]["minimise"] = evaluator.is_minimise()
        if config["evo_params"]["manager"].get("target_fitness") is not None:
            config["evo_params"]["manager"]["target_fitness"] = evaluator.get_target_fitness()
        solver = create_solver(config["evo_params"]["solver"])
        if "popsize" not in config["evo_params"]["solver"]:
            config["evo_params"]["solver"]["popsize"] = solver.popsize
        
        # Configure Evolution Manager object
        manager = EvoManager(solver, evaluator, results_path=results_path, **config["evo_params"]["manager"])

    elif manager_type == "evosax":
        from evo.manager_evosax import EvoManager, ProblemWrapper
        import evosax.algorithms as algo
        import jax.numpy as jnp
        # Configure SNN Evaluator object
        evaluator = ProblemWrapper(
            SNN_Evaluator(
                params=config
            )
        )
        # Configure Evolution Solver object
        ndim = evaluator.num_dims
        solver_class = getattr(algo, config["evo_params"]["solver"].get("type", "CMA_ES"))
        popsize = config["evo_params"]["solver"].get("popsize")
        solver = solver_class(population_size=popsize, solution=jnp.zeros(ndim, dtype=jnp.float32))
        # Configure Evolution Manager object
        manager = EvoManager(solver, evaluator, results_path=results_path, **config["evo_params"]["manager"])

    # Re-insert the manager type into the config
    config["evo_params"]["manager"]["type"] = manager_type

    # Save a copy of configuration used
    with open(results_path / "config.yaml", "w") as f:
        yaml.dump(config, f, sort_keys=False)

    # Begin experiment
    manager.run()

    # Plot fitness
    if hasattr(evaluator, "_fits_indiv_file"):
        file_path = results_path / evaluator._fits_indiv_file 
        snn_plot.plot_fitness_generation(file_path, savepath=results_path / "fitness_gen.png", show=False, estimator="median", errorband=("pi", 100))

    print(f"Evolution run completed. Results saved to {results_path}")

    return results_path 


def eval(results_path: Path | str = None, *, config_path: str | Path = None, num_steps: int = None, num_evals: int = 10, 
         rule_id: int = 1, learning_rule: LearningRule = None,
         save_plots: bool = False, show_plots: bool = False, save_results: bool = False,
         verbose: bool = True, return_evaluator: bool = False) -> Tuple[float, float] | SNN_Evaluator:

    if results_path is not None:
        if not isinstance(results_path, Path):
            results_path = Path(results_path)
    else:
        results_path = None

    if config_path is None:
        if results_path is not None:
            config_path = results_path / "config.yaml"
        else:
            raise ValueError("results_path must be specified if config_path is not specified.")
    else:
        config_path = config_path

    # Read config path
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    T = config["num_sim_steps"] if num_steps is None else num_steps
    if learning_rule is None:
        rule_id_name = f"best_rule_{rule_id:02d}.txt"
        if not (results_path / rule_id_name).exists():
            raise FileNotFoundError(f"Rule file {rule_id_name} not found in {results_path}. Please run the evolution first.")
        
        # Load the best ANN learning rule
        lrule = read_ANN_Rule(results_path / rule_id_name, config_path=results_path / "config.yaml")
        prefix = f"eval_rule_{rule_id:02d}"
    else:
        assert isinstance(learning_rule, LearningRule), f"{type(learning_rule)} is not a LearningRule object."
        lrule = learning_rule

    # spikegen_cls = getattr(spkgen, config["spikegen_params"].pop("class", "BinaryClassGenerator"))
    # spikegen = spikegen_cls(input_size=config["snn_params"].get("input_size"), **config["spikegen_params"])
    # snn = SNN(learning_rule=arule, **config["snn_params"])

    # # Backward compatbility of fitness_params
    # if "fitness_params" not in config:
    #     config["fitness_params"] = {}
    #     if "fitness_type" in config["decoder_params"]:
    #         config["fitness_params"]["type"] = config["decoder_params"].pop("fitness_type")

    # # decoder_type = config["decoder_params"].pop("type", "final")
    # # fitnessor_type = config["fitness_params"].pop("type", "accuracy")
    # simulator = SNNSimulator(snn, spikegen, record_membrane=True, record_spikes=True, record_traces=True, record_weights=True,
    #                          params=config,
    #                         #  decoder_type=decoder_type, decoder_params=config["decoder_params"],
    #                         #  fitnessor_type=fitnessor_type, fitnessor_params=config["fitness_params"]
    #                          )

    evaluator = SNN_Evaluator(
        params=config,
        record_info=True,
        learning_rule=lrule,
        log_info=False
    )

    evaluator.setup_generation(gen_count=0, num_sets=num_evals)
    fitnesses = evaluator.evaluate(num_trials=num_evals, return_fitness_list=True)
    if save_results:
        with open(results_path / f"{prefix}_eval_result.csv", "w") as f:
            f.write("trial,fitness\n")
            for i, fitness in enumerate(fitnesses):
                f.write(f"{i},{fitness}\n")
    
    # fits = []
    # for _ in range(num_evals):
    #     simulator.reset()
    #     simulator.run(T)
    #     fitness = simulator.get_fitness()
    #     fits.append(fitness)

    mean_fts = np.mean(fitnesses)
    std_fts = np.std(fitnesses)

    if verbose:
        fitness_type = config.get("fitnessor_params", {}).get("type", "unknown")
        print(f"Mean fitness (Type: {fitness_type}): {mean_fts:.2f} +/- {std_fts:.2f} SD ({num_evals} evaluations)")

    # Plotting
    if save_plots or show_plots:
        evaluator.setup_trial(trial_count=0)
        simulator = evaluator.simulator
        simulator.reset()
        simulator.run(T)
        fitness = simulator.get_fitness()
        snn_plot.plot_spikes(simulator, x_min=T-100, x_max=T, x_eps=2, savepath=Path(results_path, f"{prefix}_spikes.png") if save_plots else None, show=show_plots)
        snn_plot.plot_membranes(simulator, x_min=T-100, x_max=T, plot_inputs=False, x_scale=20, y_scale=7, 
                                savepath=Path(results_path, f"{prefix}_membranes.png") if save_plots else None, show=show_plots)
        # snn_plot.plot_weights(simulator, div=10, savepath=Path(results_path, f"{prefix}_weights.png"), show=False)
        snn_plot.plot_weight_over_time(simulator, savepath=Path(results_path, f"{prefix}_weight_over_time.png") if save_plots else None, show=show_plots)
        snn_plot.plot_weight_heatmap(simulator, savepath=Path(results_path, f"{prefix}_weight_heatmap.png") if save_plots else None, show=show_plots, log_scale=False, t_range=500)
        if simulator.record_eligibility:
            snn_plot.plot_eligibility_traces(simulator, savepath=Path(results_path, f"{prefix}_eligibility_traces.png") if save_plots else None, show=show_plots)
        snn_plot.plot_intermediate_fitness(simulator, savepath=Path(results_path, f"{prefix}_intermediate_fitness.png") if save_plots else None, show=show_plots)

    if not return_evaluator:
        return mean_fts, std_fts
    else:
        # Return the evaluator object if requested
        return evaluator
    # return mean_fts, std_fts

if __name__ == "__main__":
    # Argument parser
    argparser = argparse.ArgumentParser(description="Run Evolutionary Learning Rule Experiment")
    argparser.add_argument("--config", "-c", type=str, default="binary_es_v2.yaml", help="Path to the configuration file")
    argparser.add_argument("--override", "-o", type=str, nargs="*", help="Override specific config values (e.g., snn_params.neuron_params.tau_mem=0.05)")
    argparser.add_argument("--parent", "-p", type=str, default=None, help="Parent run directory to save results in. Useful for running series of related results.")
    argparser.add_argument("--num_evals", "-e", type=int, default=10, help="Number of evaluation trials to run for the best rule at the end.")
    args = argparser.parse_args()

    # Parse overrides
    config_overrides = parse_config_overrides(args.override) if args.override else {}

    # Run main function
    results_path = main(config_file=args.config, config_overrides=config_overrides, parent_run=args.parent)
    # Evaluation of best solution
    eval(results_path, save_plots=True, save_results=True, num_evals=args.num_evals, verbose=True)