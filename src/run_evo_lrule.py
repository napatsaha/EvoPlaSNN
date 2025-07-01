
import time
from typing import Tuple
import yaml
from pathlib import Path
import numpy as np

from evo.manager import EvoManager
from snn.eval import SNN_Evaluator
# from evo.es import EvolutionStrategy
from evo.utils import create_solver

from snn.plot import plot_weight_over_time, plot_weights, plot_spikes, plot_membranes

def main(config_file: str | Path) -> Path:
    ROOT = Path(__file__).parent.parent

    # Default config file
    config_path = Path(ROOT, "config", config_file)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Directory to save new results
    results_path = Path(ROOT, "results", "binary_es", time.strftime("%y-%m-%d_%H-%M"))
    results_path.mkdir(parents=True, exist_ok=True)

    log_file = Path(results_path, "best.log")

    # Configure SNN Evaluator object
    evaluator = SNN_Evaluator(
        params=config
        # num_simulation_steps=config["num_sim_steps"],
        # snn_params=config["snn_params"],
        # spikegen_params=config["spikegen_params"],
        # arule_params=config["arule_params"],
        # decoder_params=config["decoder_params"],
        # fitnessor_params=config["fitness_params"]
    )
    # Configure Evolution Solver object
    ndim = evaluator.get_parameter_size()
    is_minimise = evaluator.is_minimise()
    # config["evo_params"]["solver"].pop("ndim", None)  # Remove ndim from solver config if it exists
    config["evo_params"]["solver"]["ndim"] = ndim
    config["evo_params"]["solver"]["minimise"] = is_minimise
    # solver = EvolutionStrategy(**config["evo_params"]["solver"])
    solver = create_solver(config["evo_params"]["solver"])
    # Configure Evolution Manager object
    manager = EvoManager(solver, evaluator, log_file=log_file, **config["evo_params"]["manager"])

    # Save a copy of configuration used
    with open(results_path / "config.yaml", "w") as f:
        yaml.dump(config, f, sort_keys=False)

    # Begin experiment
    manager.run()

    return results_path 


def eval(results_path: Path | str, num_steps: int = None, rule_id: int = 1, num_evals: int = 10, save_plots: bool = False, verbose: bool = True) -> Tuple[float, float]:

    if not isinstance(results_path, Path):
        results_path = Path(results_path)

    with open(results_path / "config.yaml", "r") as f:
        config = yaml.safe_load(f)

    T = config["num_sim_steps"] if num_steps is None else num_steps
    rule_id_name = f"best_rule_{rule_id:02d}.txt"
    if not (results_path / rule_id_name).exists():
        raise FileNotFoundError(f"Rule file {rule_id_name} not found in {results_path}. Please run the evolution first.")
    
    # # Load the best ANN learning rule
    # arule = read_ANN_Rule(results_path / rule_id_name, config_path=results_path / "config.yaml")

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
        record_info=True
    )
    simulator = evaluator.simulator
    
    fits = []
    for _ in range(num_evals):
        simulator.reset()
        simulator.run(T)
        fitness = simulator.get_fitness()
        fits.append(fitness)

    mean_fits = sum(fits) / len(fits)
    std_fits = np.std(fits)

    if verbose:
        fitness_type = config.get("fitnessor_params", {}).get("type", "unknown")
        print(f"Mean fitness (Type: {fitness_type}): {mean_fits:.2f} ({num_evals} evaluations)")

    # Plotting
    if save_plots:
        simulator.reset()
        simulator.run(T)
        fitness = simulator.get_fitness()
        prefix = f"eval_rule_{rule_id:02d}"
        plot_spikes(simulator, x_min=T-100, x_max=T, x_eps=2, savepath=Path(results_path, f"{prefix}_spikes.png"), show=False)
        plot_membranes(simulator, x_min=T-100, x_max=T, plot_inputs=False, col_width=20, row_height=7, savepath=Path(results_path, f"{prefix}_membranes.png"), show=False)
        plot_weights(simulator, div=10, savepath=Path(results_path, f"{prefix}_weights.png"), show=False)
        plot_weight_over_time(simulator, savepath=Path(results_path, f"{prefix}_weight_over_time.png"), show=False)

    return mean_fits, std_fits

if __name__ == "__main__":
    results_path = main(config_file="binary_es_v2.yaml")
    # Evaluation of best solution
    eval(results_path, save_plots=True)