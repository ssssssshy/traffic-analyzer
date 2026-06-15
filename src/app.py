import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
from config import DB_PATH, DATA_DIR
import cv2
import numpy as np
import math
from database import get_setting, update_setting

# Настройка страницы Streamlit
st.set_page_config(page_title="Shawarma Traffic Analytics", layout="wide")

st.title("📊 Аналитика пешеходного трафика для точки")
st.markdown(
    "Данные собираются автоматически с помощью компьютерного зрения (YOLOv8 + ByteTrack)"
)


# --- ФУНКЦИИ ОБРАТНОГО ВЫЗОВА (CALLBACKS) ---
def on_setting_change(key_name: str):
    """Мгновенно сохраняет измененный ползунок в БД."""
    update_setting(key_name, st.session_state[key_name])


# --- БОКОВАЯ ПАНЕЛЬ НАСТРОЕК ПРОСТРАНСТВА ---
st.sidebar.header("⚙️ Пространственные настройки (Транзит)")

current_pos_x = get_setting("line_position_x", 0.5)
current_width = get_setting("line_width", 0.15)
current_angle = get_setting("line_angle_v", 0.0)

# 1. Сдвиг по горизонтали
st.sidebar.slider(
    "1️⃣ Смещение зоны по горизонтали (%)",
    min_value=0.1,
    max_value=0.9,
    value=float(current_pos_x),
    step=0.02,
    key="line_position_x",
    on_change=on_setting_change,
    args=("line_position_x",),
)

# 2. Ширина коридора
st.sidebar.slider(
    "2️⃣ Ширина коридора детекции",
    min_value=0.02,
    max_value=0.40,
    value=float(current_width),
    step=0.01,
    key="line_width",
    on_change=on_setting_change,
    args=("line_width",),
)

# 3. Наклон вертикальной оси
st.sidebar.slider(
    "3️⃣ Наклон вертикальных линий (градусы)",
    min_value=-30.0,
    max_value=30.0,
    value=float(current_angle),
    step=1.0,
    key="line_angle_v",
    on_change=on_setting_change,
    args=("line_angle_v",),
)

st.sidebar.markdown("---")
st.sidebar.header("➕ Ручное добавление")
col_manual1, col_manual2 = st.sidebar.columns(2)
with col_manual1:
    if st.button("➕ ВЛЕВО (IN)"):
        from database import log_pedestrian
        log_pedestrian(track_id=0, direction="IN")
        st.toast("Добавлен проход ВЛЕВО (IN)")
        st.rerun()
with col_manual2:
    if st.button("➕ ВПРАВО (OUT)"):
        from database import log_pedestrian
        log_pedestrian(track_id=0, direction="OUT")
        st.toast("Добавлен проход ВПРАВО (OUT)")
        st.rerun()


# --- ВЕРХНЯЯ ЗОНА: ИНТЕРАКТИВНЫЙ ОТЛАДЧИК ЛИНИИ ---
st.subheader("👁️ Настройка геометрии кадра в реальном времени")


@st.cache_data(show_spinner=False)
def get_preview_frame(video_filename="IMG_1686.MOV"):
    """Загружает ровно один первый кадр видео для превью из папки data."""
    video_path = DATA_DIR / video_filename
    cap = cv2.VideoCapture(str(video_path))
    success, frame = cap.read()
    cap.release()
    if success:
        return frame
    return None


# Берем кадр из вашего видео
preview_frame = get_preview_frame()

if preview_frame is not None:
    # Копируем кадр, чтобы не портить оригинал в кэше
    draw_frame = preview_frame.copy()
    height, width = draw_frame.shape[:2]

    # Берем значения ползунков напрямую из session_state (или текущие из базы)
    p_x = st.session_state.get("line_position_x", current_pos_x)
    w_w = st.session_state.get("line_width", current_width)
    a_g = st.session_state.get("line_angle_v", current_angle)

    # Математика линий (такая же, как в main.py)
    angle_rad = math.radians(a_g)
    tan_a = math.tan(angle_rad)
    center_x = int(width * p_x)
    half_height = height / 2
    half_w_thick = (width * w_w) / 2

    def get_x(y, offset):
        return int(center_x + offset + (y - half_height) * tan_a)

    # Строим полигон коридора
    pts = np.array(
        [
            [get_x(0, -half_w_thick), 0],
            [get_x(0, half_w_thick), 0],
            [get_x(height, half_w_thick), height],
            [get_x(height, -half_w_thick), height],
        ],
        np.int32,
    )

    # Накладываем полупрозрачную неоновую заливку зоны
    overlay = draw_frame.copy()
    cv2.fillPoly(overlay, [pts], (0, 255, 255))
    cv2.addWeighted(overlay, 0.3, draw_frame, 0.7, 0, draw_frame)

    # Рисуем четкие границы
    cv2.line(
        draw_frame,
        (get_x(0, -half_w_thick), 0),
        (get_x(height, -half_w_thick), height),
        (0, 165, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.line(
        draw_frame,
        (get_x(0, half_w_thick), 0),
        (get_x(height, half_w_thick), height),
        (0, 165, 255),
        3,
        cv2.LINE_AA,
    )

    # Конвертируем BGR в RGB для корректного отображения в браузере
    draw_frame_rgb = cv2.cvtColor(draw_frame, cv2.COLOR_BGR2RGB)

    # Выводим картинку в дашборд
    st.image(
        draw_frame_rgb,
        caption="Желтая зона — коридор фиксации транзита. Люди должны пересекать его полностью (слева направо или справа налево).",
        use_container_width=True,
    )
else:
    st.warning(
        "Файл `IMG_1686.MOV` не найден в корне проекта. Загрузите его, чтобы увидеть превью камеры."
    )


st.markdown("---")


# --- ДАЛЬШЕ ИДЕТ ВАШ СТАНДАРТНЫЙ БЛОК ОТРИСОВКИ KPI И ГРАФИКОВ ---
def load_data_from_db():
    try:
        conn = sqlite3.connect(DB_PATH)
        query = "SELECT timestamp, track_id, direction FROM traffic"
        df_raw = pd.read_sql_query(query, conn)
        conn.close()
        if not df_raw.empty:
            df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])
        return df_raw
    except Exception as e:
        st.error(f"Ошибка подключения к базе данных: {e}")
        return pd.DataFrame()


df = load_data_from_db()

if df.empty:
    st.info(
        "В базе данных пока нет записей о прохожих. Настройте коридор выше и запустите `main.py`!"
    )
else:
    # Метрики
    total_count = df["track_id"].nunique()
    in_count = df[df["direction"] == "IN"].shape[0]
    out_count = df[df["direction"] == "OUT"].shape[0]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Всего прохожих", total_count)
    with col2:
        st.metric("Транзит ВЛЕВО (IN)", in_count)
    with col3:
        st.metric("Транзит ВПРАВО (OUT)", out_count)
    with col4:
        st.metric("Прогноз заказов (1.2%)", int(in_count * 0.012))

    # Графики
    df["Hour"] = df["timestamp"].dt.hour
    hourly_traffic = (
        df.groupby(["Hour", "direction"]).size().reset_index(name="Количество людей")
    )

    left_column, right_column = st.columns(2)
    with left_column:
        st.subheader("🕒 Распределение по часам")
        fig_hourly = px.bar(
            hourly_traffic,
            x="Hour",
            y="Количество людей",
            color="direction",
            barmode="group",
        )
        st.plotly_chart(fig_hourly, use_container_width=True)
    with right_column:
        st.subheader("📈 Интенсивность во времени")
        df_resampled = (
            df.set_index("timestamp")
            .resample("15Min")
            .size()
            .reset_index(name="Количество")
        )
        fig_line = px.line(
            df_resampled, x="timestamp", y="Количество", line_shape="spline"
        )
        st.plotly_chart(fig_line, use_container_width=True)

    st.subheader("📋 Последние добавленные прохожие")
    st.dataframe(
        df.sort_values(by="timestamp", ascending=False).head(20),
        use_container_width=True,
    )
