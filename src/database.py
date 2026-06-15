import sqlite3
from datetime import datetime
from config import DB_PATH


def init_db():
    """Создает таблицу в БД, если её еще нет."""
    # Убедимся, что папка для БД существует
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Создаем таблицу логов трафика
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS traffic (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,    -- Дата и время пересечения (YYYY-MM-DD HH:MM:SS)
            track_id INTEGER NOT NULL,  -- Уникальный ID человека от трекера YOLO
            direction TEXT NOT NULL     -- Направление: 'IN' (к точке) или 'OUT' (от точки)
        )
    """)

    conn.commit()
    conn.close()


def log_pedestrian(track_id: int, direction: str):
    """Записывает одного прошедшего человека в базу."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute(
        "INSERT INTO traffic (timestamp, track_id, direction) VALUES (?, ?, ?)",
        (current_time, track_id, direction),
    )

    conn.commit()
    conn.close()


if __name__ == "__main__":
    # Тестовый запуск для проверки создания БД
    init_db()
    print(f"База данных успешно инициализирована по пути: {DB_PATH}")
