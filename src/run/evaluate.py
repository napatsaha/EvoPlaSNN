import pandas as pd
import snn.plot as snn_plot
import yaml
from common.base import Evaluator, LearningRule
from lrule.utils import read_learning_rule
from rl.eval import RL_Evaluator


from pathlib import Path
from typing import List, Tuple


def _plot_simulation_results(simulator: 'SNNSimulator', results_path, prefix, save_plots, show_plots, plot_params, default_plot_flag):
    # Plot spike raster
    if plot_params.get("plot_spikes", default_plot_flag):
        try:
            snn_plot.plot_spikes(simulator, x_eps=2, x_range=200,
                                savepath=Path(results_path, f"{prefix}_spikes.png") if save_plots else None, show=show_plots)
        except Exception as e:
            print(f"Error plotting spikes: {e}")
    # Plot membranes and threshold
    if plot_params.get("plot_membranes", default_plot_flag):
        try:
            snn_plot.plot_membranes(simulator, plot_inputs=False, x_scale=0.3, y_scale=3, layout=None, x_range=200,
                                    savepath=Path(results_path, f"{prefix}_membranes.png") if save_plots else None, show=show_plots)
        except Exception as e:
            print(f"Error plotting membranes: {e}")
    # Plot static weight at end of simulation
    if plot_params.get("plot_weights", default_plot_flag):
        try:
            snn_plot.plot_weights(simulator, env=simulator.env, bounded_weights=False, y_scale=1.0, x_scale=1.0,
                                savepath=Path(results_path, f"{prefix}_weights.png") if save_plots else None, show=show_plots)
        except Exception as e:
            print(f"Error plotting weights: {e}")
    # Plot weight changes as line plots
    if plot_params.get("plot_weight_over_time", default_plot_flag):
        try:
            snn_plot.plot_weight_over_time(simulator, synapse_layer=0,
                                        savepath=Path(results_path, f"{prefix}_weight_over_time.png") if save_plots else None, show=show_plots)
        except Exception as e:
            print(f"Error plotting weight over time: {e}")
    # Plot weight changes as horizontal heatmap
    # try:
    #     snn_plot.plot_weight_heatmap(simulator, log_scale=False, t_range=500, synapse_layer=0,
    #                                  savepath=Path(results_path, f"{prefix}_weight_heatmap.png") if save_plots else None, show=show_plots)
    # except Exception as e:
    #     print(f"Error plotting weight heatmap: {e}")
    # Plot environment weights: all actions
    if plot_params.get("plot_env_weight_actions", default_plot_flag):
        try:
            snn_plot.plot_env_weight_actions(simulator,
                                                savepath=Path(results_path, f"{prefix}_env_weight_actions.png") if save_plots else None, show=show_plots)
        except Exception as e:
            print(f"Error plotting environment weight actions: {e}")
    # Plot environment weights: greedy actions
    if plot_params.get("plot_env_weight_greedy", default_plot_flag):
        try:
            snn_plot.plot_env_weight_greedy(simulator,
                                            savepath=Path(results_path, f"{prefix}_env_weight_greedy.png") if save_plots else None, show=show_plots)
        except Exception as e:
            print(f"Error plotting environment weight greedy: {e}")
    # Plot pre-before-post eligibility traces
    if plot_params.get("plot_eligibility_pre", default_plot_flag) and simulator.record_eligibility_pre:
        try:
            snn_plot.plot_eligibility_traces(simulator, etype="pre", synapse_layer=0,
                                                savepath=Path(results_path, f"{prefix}_eligibility_pre_traces.png") if save_plots else None, show=show_plots)
        except Exception as e:
            print(f"Error plotting eligibility pre-before-post traces: {e}")
    # Plot post-before-pre eligibility traces
    if plot_params.get("plot_eligibility_post", default_plot_flag) and simulator.record_eligibility_post:
        try:
            snn_plot.plot_eligibility_traces(simulator, etype="post", synapse_layer=0,
                                            savepath=Path(results_path, f"{prefix}_eligibility_post_traces.png") if save_plots else None, show=show_plots)
        except Exception as e:
            print(f"Error plotting eligibility post-before-pre traces: {e}")
    # Plot STDP eligibility traces
    if plot_params.get("plot_eligibility_stdp", default_plot_flag) and simulator.record_eligibility_stdp:
        try:
            snn_plot.plot_eligibility_traces(simulator, etype="stdp", synapse_layer=0,
                                            savepath=Path(results_path, f"{prefix}_eligibility_stdp_traces.png") if save_plots else None, show=show_plots)
        except Exception as e:
            print(f"Error plotting eligibility STDP traces: {e}")
    # Plot Custom (rule-derived) eligibility traces
    if plot_params.get("plot_eligibility_custom", default_plot_flag) and simulator.record_eligibility_custom:
        try:
            snn_plot.plot_eligibility_traces(simulator, etype="custom", synapse_layer=0,
                                            savepath=Path(results_path, f"{prefix}_eligibility_custom_traces.png") if save_plots else None, show=show_plots)
        except Exception as e:
            print(f"Error plotting eligibility Custom traces: {e}")
    # Plot intermediate fitness within simulation
    if plot_params.get("plot_intermediate_fitness", default_plot_flag):
        try:
            snn_plot.plot_intermediate_fitness(simulator, window_size=20, plot_exploration=True, figsize=(20, 10),
                                            savepath=Path(results_path, f"{prefix}_intermediate_fitness.png") if save_plots else None, show=show_plots)
        except Exception as e:
            print(f"Error plotting intermediate fitness: {e}")


def evaluate_and_plot(results_path: Path | str = None, *, config_path: str | Path = None, num_steps: int = None, num_evals: int = 10,
         rule_id: int = 1, learning_rule: LearningRule = None,
         save_plots: bool = False, show_plots: bool = False, save_results: bool = False, eval_results: bool = True,
         plot_params: dict = None, default_plot_flag: bool = True,
         verbose: bool = True, return_evaluator: bool = False) -> Tuple[float, float] | Evaluator:

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

    # Plot params
    if plot_params is None:
        plot_params: dict = config.get("plot_params", {})
    if "plot_simulation" not in plot_params:
        plot_params["plot_simulation"] = plot_params.get("plot_spikes", default_plot_flag) or \
            plot_params.get("plot_membranes", default_plot_flag) or \
            plot_params.get("plot_weights", default_plot_flag) or \
            plot_params.get("plot_weight_over_time", default_plot_flag) or \
            plot_params.get("plot_env_weight_actions", default_plot_flag) or \
            plot_params.get("plot_env_weight_greedy", default_plot_flag) or \
            plot_params.get("plot_eligibility_pre", default_plot_flag) or \
            plot_params.get("plot_eligibility_post", default_plot_flag) or \
            plot_params.get("plot_eligibility_stdp", default_plot_flag) or \
            plot_params.get("plot_eligibility_custom", default_plot_flag) or \
            plot_params.get("plot_intermediate_fitness", default_plot_flag)

    # T = config["num_sim_steps"] if num_steps is None else num_steps
    if learning_rule is None:
        rule_id_name = f"best_rule_{rule_id:02d}.txt"
        if not (results_path / rule_id_name).exists():
            raise FileNotFoundError(f"Rule file {rule_id_name} not found in {results_path}. Please run the evolution first.")

        # Load the best ANN learning rule
        rule = read_learning_rule(results_path / rule_id_name, config_path=config_path)
        prefix = f"eval_rule-{rule_id:02d}"
    else:
        assert isinstance(learning_rule, LearningRule), f"{type(learning_rule)} is not a LearningRule object."
        rule = learning_rule
        prefix = f"eval_custom_rule"

    # For compatibility with newer versions of config without "manager" or "evaluator" in "evo_params"
    evaluator_params = config.get("evaluator_params") if "evaluator_params" in config else \
        config["evo_params"]["evaluator"]
    if "evo_params" in config and "manager" in config["evo_params"]:
        manager_params = config["evo_params"]["manager"]
    elif "evo_params" in config:
        manager_params = config["evo_params"]
    
    # Prevent logging
    evaluator_params.update(
        {"log_level": 0,
         "record_inter_fitness": False}
    )
    multiple_evaluators = manager_params.get("multiple_evaluators", False)

    evaluator: Evaluator = None
    evaluators: List[RL_Evaluator] = None
    if not multiple_evaluators:
        evaluator: Evaluator = RL_Evaluator(
            params=config,
            record_info=True,
            **evaluator_params
        )
    else:
        envs_params: dict = config.get("envs_params", {})
        evaluators = []
        num_envs = len(envs_params["files"])

        for file in envs_params.get("files", []):
            with open(file) as f:
                env_config = yaml.safe_load(f)
                env_config = env_config.get("env_params", env_config)
            evaluator: Evaluator = RL_Evaluator(
                params=config,
                env_params=env_config,
                record_info=True,
                **evaluator_params
            )
            evaluators.append(evaluator)

    if eval_results:
        if multiple_evaluators:
            for i, evaluator in enumerate(evaluators):
                fts_list, avg_fts, std_fts, behv = evaluator.evaluate(genome=rule, num_trials=num_evals)
                prefix_i = prefix + "_" + f"env-{i+1:02d}"
                if save_results:
                    with open(results_path / f"{prefix_i}_eval_result.csv", "w") as f:
                        f.write("trial,fitness\n")
                        for i, fitness in enumerate(fts_list):
                            f.write(f"{i},{fitness}\n")
        else:
            fts_list, avg_fts, std_fts, behv = evaluator.evaluate(genome=rule, num_trials=num_evals)
            if save_results:
                with open(results_path / f"{prefix}_eval_result.csv", "w") as f:
                    f.write("trial,fitness\n")
                    for i, fitness in enumerate(fts_list):
                        f.write(f"{i},{fitness}\n")

        if verbose:
            fitness_type = config.get("fitnessor_params", {}).get("type", "unknown")
            print(f"Mean fitness (Type: {fitness_type}): {avg_fts:.2f} +/- {std_fts:.2f} SD ({num_evals} evaluations)")

    # Plotting
    if save_plots or show_plots:

        # Plot fitness
        if plot_params.get("plot_fitness_gen", default_plot_flag):
            try:
                if (results_path / "fitness_per_indiv.csv").exists():
                    snn_plot.plot_fitness_generation(results_path / "fitness_per_indiv.csv", savepath=results_path / "offspring_fitness_gen.png", show=False,
                                                    estimator="median", errorband=("pi", 50))
                elif multiple_evaluators and bool(manager_params.get("log_indiv", False)):
                    res_list = []
                    for fts_file in results_path.glob("fitness_per_indiv*"):
                        res_list.append(pd.read_csv(fts_file))
                    res = pd.concat(res_list, axis=0, keys=[*range(num_envs)], names=["envs", ""])
                    res = res.reset_index(level="envs").reset_index(drop=True)
                    snn_plot.plot_fitness_generation(res=res, hue_var="envs", estimator="median", errorband=("pi", 50), merge_avg=True, linecolor_est="black",
                                                    run_name=results_path.stem,
                                                    savepath=results_path / "offspring_fitness_gen.png", show=False)
                else:
                    raise AssertionError("No file 'fitness_per_indiv.csv' or 'fitness_per_indiv_env-**.csv' recorded")
            except Exception as e:
                print(f"Error plotting offspring fitness: {e}")
        # Plot solution fitness
        if plot_params.get("plot_solution_gen", default_plot_flag):
            try:
                if (results_path / "solutions.csv").exists():
                    run_name = results_path.stem
                    # Plot solution global fitness
                    sols_df = snn_plot.plot_solution_generation(results_path / "solutions.csv", var="global_fitness", comment=f"Run: {run_name}",
                                                                point_cmap="husl", estimator="median", errorband=("pi", 75),
                                                                show=False, savepath=results_path / "solution_fitness_gen.png")
                    # Plot solution novelty distance
                    snn_plot.plot_solution_generation(df=sols_df, var="novelty_dist", comment=f"Run: {run_name}", point_cmap="husl",
                                                    estimator="median", errorband=("pi", 50),
                                                    show=False, savepath=results_path / "solution_novelty_gen.png")
                    snn_plot.plot_solution_generation(df=sols_df, var="local_fitness", comment=f"Run: {run_name}", point_cmap="husl",
                                                    estimator="mean", errorband=("se", 3),
                                                    show=False, savepath=results_path / "solution_local_gen.png")
                    snn_plot.plot_solution_generation(df=sols_df, var="rank", comment=f"Run: {run_name}", point_cmap="husl",
                                                    estimator="median", errorband=("pi", 50),
                                                    show=False, savepath=results_path / "solution_rank_gen.png")
                else:
                    raise AssertionError("No file 'solutions.csv' recorded")
            except Exception as e:
                print(f"Error plotting solution fitness: {e}")

        # Plot Learning Rule Response for each rule in save_best
        if plot_params.get("plot_learning_rule", default_plot_flag):
            try:
                num_save_best = manager_params.get("save_best", 0)
                for rule_id in range(1, num_save_best+1):
                    rule_id_name = f"best_rule_{rule_id:02d}.txt"
                    if not (results_path / rule_id_name).exists():
                        raise FileNotFoundError(f"Rule file {rule_id_name} not found in {results_path}. Please run the evolution first.")
                    # Load the best ANN learning rule
                    rule_i = read_learning_rule(results_path / rule_id_name, config_path=config_path)
                    prefix_i = f"eval_rule_{rule_id:02d}"
                    snn_plot.plot_learning_rule(rule_i,
                                                savepath=Path(results_path, f"{prefix_i}_learning_rule.png") if save_plots else None, show=show_plots)
            except Exception as e:
                print(f"Error plotting learning rule: {e}")

        # Plot which rely on simulation
        if plot_params["plot_simulation"]: # To prevent unnecessary evaluation if no subplots requiring trial simulation are needed
            if multiple_evaluators:
                for i, evaluator in enumerate(evaluators):
                    prefix_i = prefix + "_" + f"env-{i+1:02d}"

                    simulator = evaluator.simulator
                    simulator.learning_rule = rule
                    simulator.reset()
                    simulator.run(num_steps=evaluator.max_steps, num_eps=evaluator.max_episodes)
                    fitness = simulator.get_fitness()

                    _plot_simulation_results(simulator, results_path, prefix_i, save_plots, show_plots, plot_params, default_plot_flag)
            else:
                simulator = evaluator.simulator
                simulator.learning_rule = rule
                simulator.reset()
                simulator.run(num_steps=evaluator.max_steps, num_eps=evaluator.max_episodes)
                fitness = simulator.get_fitness()

                _plot_simulation_results(simulator, results_path, prefix, save_plots, show_plots, plot_params, default_plot_flag)


    if not return_evaluator and eval_results:
        return avg_fts, std_fts
    else:
        # Return the evaluator object if requested
        return evaluator if not multiple_evaluators else evaluators
