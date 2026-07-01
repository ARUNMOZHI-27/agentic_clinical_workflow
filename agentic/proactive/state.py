
from typing import TypedDict
from datetime import datetime
import pandas as pd
import numpy as np

class WorkflowState(TypedDict, total=False):
    stay_df: pd.DataFrame
    trigger_time: datetime
    sequence: np.ndarray

    predicted_delay: float
    alert_flag: bool