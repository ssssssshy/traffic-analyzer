# Project: Traffic Analyzer

## Overview
Traffic Analyzer is a real-time pedestrian transit monitoring system. It uses computer vision (YOLOv8 + ByteTrack) to detect and track individuals crossing a defined zone in a video feed.

## Technology Stack
- **Vision Model:** YOLOv8 (via `ultralytics`)
- **Web Interface:** Streamlit
- **Database:** SQLite
- **Data Manipulation:** pandas
- **Visualization:** plotly, OpenCV

## Architecture
- `src/app.py`: Streamlit-based dashboard for real-time visualization and spatial configuration of the detection zone.
- `src/database.py`: Handles SQLite interactions for logging pedestrian transit data and storing system settings (e.g., zone dimensions).
- `src/main.py`: Primary script for running the computer vision processing pipeline.
- `src/config.py`: Centralized configuration.

## Setup & Running
- **Installation:** The project uses `uv` for dependency management.
- **Run Dashboard:** `streamlit run src/app.py`
- **Run Detection:** `python src/main.py`

## Development Conventions
- Database configurations are managed via a persistent SQLite store (`data/traffic_logs.db`).
- UI/UX interactions in Streamlit rely on callback functions (`on_setting_change`) to synchronize settings between the dashboard and the database.
