
from pathlib import Path
import sys

print(__file__)

include_dir = Path(__file__).parent
sys.path.append(str(include_dir))

import CheckRequirements
CheckRequirements.check_requirements()

from main import *

bl_info = {
    "name": "CamAlign",
    "blender": (2, 80, 0),
    "category": "Tool",
}