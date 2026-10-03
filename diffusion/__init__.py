# Modified from OpenAI's diffusion repos
#     GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
#     ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
#     IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py

from importlib import import_module

__all__ = ["DPMS", "FlowEuler", "Scheduler", "SASolverSampler"]


def __getattr__(name):
    """Load runtime exports on demand so configuration imports stay lightweight."""
    modules = {
        "DPMS": ".dpm_solver",
        "FlowEuler": ".flow_euler_sampler",
        "Scheduler": ".iddpm",
        "SASolverSampler": ".sa_sampler",
    }
    if name not in modules:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(modules[name], __name__), name)
    globals()[name] = value
    return value
