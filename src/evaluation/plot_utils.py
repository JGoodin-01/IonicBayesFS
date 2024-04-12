import os
import matplotlib.pyplot as plt
from functools import wraps

def plot_wrapper(
    figsize=(8, 6),
    xlabel="",
    ylabel="",
    scale=None,
    filename="image.svg",
    dynamic_params_func=None,
    get_image_directory=lambda: "./"
):
    def decorator(plot_func):
        @wraps(plot_func)
        def wrapper(*args, **kwargs):
            nonlocal filename  # Ensures filename can be modified by dynamic_params_func
            image_directory = get_image_directory()

            # Dynamic parameter processing
            if dynamic_params_func is not None:
                dynamic_params = dynamic_params_func(*args, **kwargs)
                dynamic_filename = dynamic_params.get("filename", filename)
            else:
                dynamic_filename = filename

            if figsize is not None:
                plt.figure(figsize=figsize)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            if scale is not None:
                plt.yscale(scale)
                plt.xscale(scale)

            plot_func(*args, **kwargs)

            if not os.path.exists(image_directory):
                os.makedirs(image_directory)
            plt.savefig(os.path.join(image_directory, dynamic_filename), format="svg")
            plt.close()
        
        return wrapper
    return decorator
