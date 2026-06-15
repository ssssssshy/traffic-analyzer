import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
from database import get_line_position, update_line_position
from config import DB_PATH

# Настройка страницы Streamlit
st.set_page_config(page_title="Shawarma Traffic Analytics", layout="wide")

st.title("📊 Аналитика пешеходного трафика для точки")
st.markdown(
    "Данные собираются автоматически с помощью компьютерного зрения (YOLOv8 + ByteTrack)"
)

# --- БОКОВАЯ ПАНЕЛЬ НАСТРОЕК ---
st.sidebar.header("⚙️ Настройки камеры")

# Читаем текущее значение из БД для инициализации ползунка
current_line_pos = get_line_position()

# Ползунок от 0.1 (верх кадра) до 0.9 (низ кадра)
new_line_pos = st.sidebar.slider(
    "Положение линии подсчета (высота %)",
    min_value=0.1,
    max_value=0.9,
    value=float(current_line_pos),
    step=0.05,
    help="0.5 — ровно посередине экрана. Чем больше значение, тем ниже опускается линия.",
)

# Если пользователь подвинул ползунок — сохраняем в БД и жестко обновляем интерфейс
if new_line_pos != current_line_pos:
    update_line_position(new_line_pos)
    st.sidebar.success(f"Линия сдвинута на {int(new_line_pos * 100)}%")
    st.rerun()  # Перезапускает интерфейс, чтобы весь проект синхронизировался


def load_data_from_db():
    """Загрузка сырых данных из БД и превращение в DataFrame."""
    try:
        conn = sqlite3.connect(DB_PATH)
        query = "SELECT timestamp, track_id, direction FROM traffic"
        df_raw = pd.read_sql_query(query, conn)
        conn.close()

        if not df_raw.empty:
            # Приводим к типу datetime для удобной группировки
            df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])
        return df_raw
    except Exception as e:
        st.error(f"Ошибка подключения к базе данных: {e}")
        return pd.DataFrame()


# Загружаем актуальные данные
df = load_data_from_db()

if df.empty:
    st.info(
        "В базе данных пока нет записей о прохожих. Запустите `main.py`, чтобы собрать тестовый трафик!"
    )
else:
    # --- ВЕРХНИЕ МЕТРИКИ (KPI) ---
    total_count = df["track_id"].nunique()
    in_count = df[df["direction"] == "IN"].shape[0]
    out_count = df[df["direction"] == "OUT"].shape[0]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Всего прохожих", total_count)
    with col2:
        st.metric("Идут К точке (IN)", in_count)
    with col3:
        st.metric("Идут ОТ точки (OUT)", out_count)
    with col4:
        # Средняя конверсия в стрит-фуде около 1-1.5% от проходящего мимо трафика
        estimated_orders = int(in_count * 0.012)
        st.metric("Прогноз заказов (конверсия 1.2%)", estimated_orders)

    st.markdown("---")

    # --- ПОДГОТОВКА ДАННЫХ ДЛЯ ГРАФИКОВ ---
    df["Hour"] = df["timestamp"].dt.hour
    df["Date"] = df["timestamp"].dt.date

    # Группировка по часам для графика пиков
    hourly_traffic = (
        df.groupby(["Hour", "direction"]).size().reset_index(name="Количество людей")
    )

    # --- ВИЗУАЛИЗАЦИЯ ---
    left_column, right_column = st.columns(2)

    with left_column:
        st.subheader("🕒 Распределение трафика по часам (Пиковые зоны)")
        fig_hourly = px.bar(
            hourly_traffic,
            x="Hour",
            y="Количество людей",
            color="direction",
            barmode="group",
            labels={"Hour": "Час суток", "Количество людей": "Пешеходы"},
            color_discrete_map={"IN": "#00CC96", "OUT": "#EF553B"},
        )
        fig_hourly.update_layout(xaxis_type="category")
        st.plotly_chart(fig_hourly, use_container_width=True)

    with right_column:
        st.subheader("📈 Интенсивность трафика во времени")
        # Группируем по 15-минутным интервалам для плавной линии
        df_resampled = (
            df.set_index("timestamp")
            .resample("15Min")
            .size()
            .reset_index(name="Количество")
        )
        fig_line = px.line(
            df_resampled,
            x="timestamp",
            y="Количество",
            labels={"timestamp": "Время", "Количество": "Пешеходы за 15 мин"},
            line_shape="spline",
        )
        st.plotly_chart(fig_line, use_container_width=True)

    # --- ТАБЛИЦА С СЫРЫМИ ДАННЫМИ ---
    st.subheader("📋 Последние зафиксированные прохожие")
    st.dataframe(
        df.sort_values(by="timestamp", ascending=False).head(50),
        use_container_width=True,
    )
