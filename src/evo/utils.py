
from common.utils import create_solver


algorithm_dict = {
    "cma_es": "CMA_ES",
    "es": "EvolutionStrategy",
    "simple": "EvolutionStrategy",
    "mu_plus_lambda": "MuPlusLambda"
}

module_dict = {
    "cma_es": ["CMA_ES"],
    "es": ["EvolutionStrategy"],
    "mu_lambda": ["MuPlusLambda"]
}

# def create_solver(params: dict, **kwargs) -> BaseSolver:
#     """
#     Create an instance of a solver based on the provided parameters.
    
#     Args:
#         params (dict): Parameters for the solver, including 'type' and other necessary attributes.
    
#     Returns:
#         BaseSolver: An instance of the specified solver.
#     """
#     params = copy.deepcopy(params)
#     solver_type = params.pop("type", "simple").lower().replace("-", "_")
#     if solver_type not in algorithm_dict:
#         raise ValueError(f"Unknown solver type: {solver_type}")
#     solver_class_name = algorithm_dict[solver_type]
    
#     # Dynamically import the class
#     module_name = [name for name in module_dict if solver_class_name in module_dict[name]][0]
#     module_name = f"evo.{module_name}"
#     module = __import__(module_name, fromlist=[algorithm_dict[solver_type]])
    
#     solver_class = getattr(module, solver_class_name)
#     return solver_class(**params)  # Pass the parameters to the solver's constructor


if __name__ == "__main__":
    from pprint import pprint
    params = dict(
        type="cma_es",
        ndim=10,
        n_best=5,
        popsize=20,
        minimise=True,
    )
    solver = create_solver(params)
    pprint(params)
    print(solver)

    params = dict(
        type="simple",
        ndim=10,
        popsize=20,
        minimise=False,
        sigma=0.5
    )
    solver = create_solver(params)
    pprint(params)
    print(solver)

    params = dict(
        type="es",
        ndim=20,
        popsize=10,
        minimise=True,
        sigma=0.01
    )
    solver = create_solver(params)
    pprint(params)
    print(solver)