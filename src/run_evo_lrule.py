import os
import time
from snn.spikegen import BinaryClassGenerator
import yaml
from pprint import pprint
from pathlib import Path
from evo.manager import EvoManager
from snn.eval import SNN_Evaluator
from evo.es import EvolutionStrategy

from snn.plot import plot_weight_over_time, plot_weights, plot_spikes, plot_membranes
from snn import SNN, SNNSimulator

from lrule.ann import read_ANN_Rule

def main():
    ROOT = Path(__file__).parent.parent

    # Default config file
    config_path = Path(ROOT, "config", "binary_es_v1.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Directory to save new results
    results_path = Path(ROOT, "results", "binary_es", time.strftime("%y-%m-%d_%H-%M"))
    results_path.mkdir(parents=True, exist_ok=True)

    log_file = Path(results_path, "best.log")

    # Save a copy of configuration used
    with open(results_path / "config.yaml", "w") as f:
        yaml.dump(config, f, sort_keys=False)

    # Setup evolution objects
    solver = EvolutionStrategy(**config["evo_params"]["solver"])
    evaluator = SNN_Evaluator(num_simulation_steps=config["num_sim_steps"],
                            snn_params=config["snn_params"],
                            spikegen_params=config["spikegen_params"],
                            arule_params=config["arule_params"],
                            )
    manager = EvoManager(solver, evaluator, log_file=log_file, **config["evo_params"]["manager"])

    # Begin experiment
    manager.run()

    return results_path 

def eval(results_path: Path, num_steps: int = None):

    with open(results_path / "config.yaml", "r") as f:
        config = yaml.safe_load(f)

    T = config["num_sim_steps"] if num_steps is None else num_steps
    arule = read_ANN_Rule(results_path / "best_rule_01.txt", config_path=results_path / "config.yaml")

    spikegen = BinaryClassGenerator(input_size=config["snn_params"].get("input_size"), **config["spikegen_params"])
    snn = SNN(learning_rule=arule, **config["snn_params"])

    simulator = SNNSimulator(snn, spikegen, record_membrane=True, record_spikes=True, record_traces=True, record_weights=True)
    simulator.reset()
    simulator.run(T)
    accuracy = simulator.reward_manager.accuracy()
    print(f"Accuracy: {accuracy:.2f}")

    plot_spikes(simulator, x_min=T-100, x_max=T, x_eps=2, savepath=Path(results_path, "eval_rule_01_spikes.png"), show=False)
    plot_membranes(simulator, x_min=T-100, x_max=T, plot_inputs=False, col_width=20, row_height=7, savepath=Path(results_path, "eval_rule_01_membranes.png"), show=False)
    plot_weights(simulator, div=10, savepath=Path(results_path, "eval_rule_01_weights.png"), show=False)
    plot_weight_over_time(simulator, savepath=Path(results_path, "eval_rule_01_weight_over_time.png"), show=False)


if __name__ == "__main__":
    results_path = main()
    # Evaluation of best solution
    eval(results_path)