import warnings

try:
    import tensorflow
except ModuleNotFoundError:
    str = ("tensorflow not found. "
           "You can install tensorflow with 'pip install tensorflow'"
           "or add it to the requirements.txt of your project.")
    warnings.warn(str)
