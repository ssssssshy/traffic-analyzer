from pathlib import Path

# Пути к директориям проекта
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
WEIGHTS_DIR = BASE_DIR / "weight"

# Настройки модели
MODEL_NAME = "yolov8n.pt"
MODEL_PATH = WEIGHTS_DIR / MODEL_NAME

# Настройки базы данных
DB_PATH = DATA_DIR / "traffic_logs.db"

# Логика подсчета (линия пересечения)
# Коэффициент от высоты кадра (0.5 — ровно по центру экрана)
LINE_POSITION_COEFF = 0.6

# Направления движения (в зависимости от того, как стоит машина/камера)
# Например, если люди идут сверху вниз кадра — это 'IN' (к шаурме), снизу вверх — 'OUT'
