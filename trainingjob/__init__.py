from typing import Dict, Type

from .classification import *
from .training_job import *

JOB_DICT: Dict[str, Type[TrainingJob]] = {
    "commoncls": CommonClsJob
}
