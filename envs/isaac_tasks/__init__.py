from .ant import Ant
from .franka_cabinet import FrankaCabinet

# Mappings from strings to environments
isaacgym_task_map = {
    "Ant": Ant,
    "FrankaCabinet": FrankaCabinet,
}
