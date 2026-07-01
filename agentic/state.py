from typing import TypedDict, Optional
from datetime import datetime
import pandas as pd
import numpy as np

class WorkflowState(TypedDict):
    stay_id: int
    stay_df: pd.DataFrame
    index: int

    trigger_time: datetime
    episode_type: int
    time_gap_min: float
    sequence:np.ndarray
    predicted_delay: float
    deadline: datetime

    actual_delay: float
    deviation: float

    proactive_flag: bool
    reactive_flag: bool