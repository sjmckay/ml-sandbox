from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("ml_sandbox")
except PackageNotFoundError:
    __version__ = "0.0.0"


from . import models
from . import dataset
from . import utils
from . import train_eval