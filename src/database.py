import sqlite3
from config import DB_PATH


def init_db():
    """Инициализация базы данных и создание таблиц."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Таблица для логов трафика
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS traffic (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            track_id INTEGER NOT NULL,
            direction TEXT NOT NULL
        )
    """)

    # Таблица для динамических настроек (адаптивная линия)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value REAL NOT NULL
        )
    """)

    # Задаем дефолтное значение для линии (0.6), если таблица пуста
    cursor.execute(
        "INSERT OR IGNORE INTO settings (key, value) VALUES ('line_position', 0.6)"
    )

    conn.commit()
    conn.close()


def log_pedestrian(track_id: int, direction: str):
    """Запись пересечения линии пешеходом в БД."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Используем встроенную функцию SQLite datetime('now', 'localtime')
    cursor.execute(
        "INSERT INTO traffic (timestamp, track_id, direction) VALUES (datetime('now', 'localtime'), ?, ?)",
        (track_id, direction),
    )

    conn.commit()
    conn.close()


def get_line_position() -> float:
    """Читает текущее положение линии из базы для main.py."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM settings WHERE key = 'line_position'")
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else 0.6


def update_line_position(value: float):
    """Обновляет положение линии из дашборда Streamlit."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT OR REPLACE INTO settings (key, value) VALUES ('line_position', ?)",
        (value,),
    )
    conn.commit()
    conn.close()
