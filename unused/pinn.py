import torch
from src.solvers.pinn import PINNExperiment, PINN, f, df, dfdx, dfdt
from src.problems import get_problem

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def exact_solution(x, y, config) -> torch.Tensor:
    problem = get_problem(config.EXAMPLE, getattr(config, "EPSILON", 0.01))
    res = problem.exact_solution(x, y)
    if not isinstance(res, torch.Tensor):
        res = torch.tensor(res, dtype=torch.float32, device=x.device)
    return res

def exact_solution_dx(x, y, config) -> torch.Tensor:
    problem = get_problem(config.EXAMPLE, getattr(config, "EPSILON", 0.01))
    res = problem.exact_dx(x, y)
    if not isinstance(res, torch.Tensor):
        res = torch.tensor(res, dtype=torch.float32, device=x.device)
    return res

def exact_solution_dt(x, y, config) -> torch.Tensor:
    problem = get_problem(config.EXAMPLE, getattr(config, "EPSILON", 0.01))
    res = problem.exact_dy(x, y)
    if not isinstance(res, torch.Tensor):
        res = torch.tensor(res, dtype=torch.float32, device=x.device)
    return res

def exact_solution_dx2(x, y, config) -> torch.Tensor:
    problem = get_problem(config.EXAMPLE, getattr(config, "EPSILON", 0.01))
    res = problem.exact_dx2(x, y)
    if not isinstance(res, torch.Tensor):
        res = torch.tensor(res, dtype=torch.float32, device=x.device)
    return res

def exact_solution_dt2(x, y, config) -> torch.Tensor:
    problem = get_problem(config.EXAMPLE, getattr(config, "EPSILON", 0.01))
    res = problem.exact_dy2(x, y)
    if not isinstance(res, torch.Tensor):
        res = torch.tensor(res, dtype=torch.float32, device=x.device)
    return res

def shift_EJ(x, t) -> torch.Tensor:
    problem = get_problem(3, 0.01)
    res = problem.shift_function(x, t)
    if not isinstance(res, torch.Tensor):
        res = torch.tensor(res, dtype=torch.float32, device=x.device if hasattr(x, 'device') else None)
    return res

def shift_EJ_dx(x, t) -> torch.Tensor:
    problem = get_problem(3, 0.01)
    res = problem.shift_dx(x, t)
    if not isinstance(res, torch.Tensor):
        res = torch.tensor(res, dtype=torch.float32, device=x.device if hasattr(x, 'device') else None)
    return res

def shift_EJ_dt(x, t) -> torch.Tensor:
    problem = get_problem(3, 0.01)
    res = problem.shift_dy(x, t)
    if not isinstance(res, torch.Tensor):
        res = torch.tensor(res, dtype=torch.float32, device=x.device if hasattr(x, 'device') else None)
    return res

def shift_EJ_dx2(x, t) -> torch.Tensor:
    problem = get_problem(3, 0.01)
    res = problem.shift_dx2(x, t)
    if not isinstance(res, torch.Tensor):
        res = torch.tensor(res, dtype=torch.float32, device=x.device if hasattr(x, 'device') else None)
    return res

def shift_EJ_dt2(x, t) -> torch.Tensor:
    problem = get_problem(3, 0.01)
    res = problem.shift_dy2(x, t)
    if not isinstance(res, torch.Tensor):
        res = torch.tensor(res, dtype=torch.float32, device=x.device if hasattr(x, 'device') else None)
    return res

__all__ = [
    "PINNExperiment",
    "PINN",
    "f",
    "df",
    "dfdx",
    "dfdt",
    "exact_solution",
    "exact_solution_dx",
    "exact_solution_dt",
    "exact_solution_dx2",
    "exact_solution_dt2",
    "shift_EJ",
    "shift_EJ_dx",
    "shift_EJ_dt",
    "shift_EJ_dx2",
    "shift_EJ_dt2",
    "device",
]
