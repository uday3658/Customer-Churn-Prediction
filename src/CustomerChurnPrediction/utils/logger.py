import os
import sys
import logging
from datetime import datetime 
# ─────────────────────────────────────────
# Log file setup
# ─────────────────────────────────────────
LOG_DIR      = "logs"
LOG_FILE     = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

os.makedirs(LOG_DIR, exist_ok=True)

# ─────────────────────────────────────────
# Log format
# ─────────────────────────────────────────
LOG_FORMAT = "[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s"

# ─────────────────────────────────────────
# Logger setup
# ─────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),   # ✅ saves to logs/ folder
        logging.StreamHandler(sys.stdout)     # ✅ prints to terminal
    ]
)

logger = logging.getLogger("CustomerChurnPrediction")