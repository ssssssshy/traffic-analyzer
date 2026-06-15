import sqlite3
from config import DB_PATH


def init_db():
    """Инициализация базы данных и создание таблиц."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS traffic (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            track_id INTEGER NOT NULL,
            direction TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value REAL NOT NULL
        )
    """)

    # Инициализируем 3 плоскости управления
    cursor.execute(
        "INSERT OR IGNORE INTO settings (key, value) VALUES ('line_position', 0.6)"
    )
    cursor.execute(
        "INSERT OR IGNORE INTO settings (key, value) VALUES ('line_thickness', 0.10)"
    )
    cursor.execute(
        "INSERT OR IGNORE INTO settings (key, value) VALUES ('line_angle', 0.0)"
    )  # Новый параметр: угол 0° (горизонтально)

    conn.commit()
    conn.close()


def log_pedestrian(track_id: int, direction: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO traffic (timestamp, track_id, direction) VALUES (datetime('now', 'localtime'), ?, ?)",
        (track_id, direction),
    )
    conn.commit()
    conn.close()


def get_setting(key: str, default: float) -> float:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM settings WHERE key = ?", (key,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else default


def update_setting(key: str, value: float):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, value)
    )
    conn.commit()
    conn.close()
