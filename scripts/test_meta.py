import sys
from pathlib import Path

PROJECT_ROOT = Path(r"E:\585project")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data.chexpert_loader import CheXpertMeta

csv_path = r"E:\CS585-Project\data\raw\chexpert\train.csv"

meta = CheXpertMeta(csv_path)
meta.show_basic_info()
meta.show_columns()
meta.target_summary()