import cv2
import torch
import numpy as np
import math
from typing import Union
from ultralytics import YOLO
from database import log_pedestrian, init_db, get_setting


def run_analytics(video_source: Union[str, int] = "data/IMG_1686.MOV"):
    init_db()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Загружаем модель
    model = YOLO("yolov8n.pt").to(device)

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Не удалось открыть видео: {video_source}")
        return

    track_history = {}
    track_states = {}  # Состояния: 'LEFT', 'RIGHT', 'COUNTED'
    counted_ids = set()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Считываем настройки горизонтального транзита из БД
        pos_x_coeff = get_setting("line_position_x", 0.5)
        width_coeff = get_setting("line_width", 0.15)
        angle_deg = get_setting("line_angle_v", 0.0)

        height, width = frame.shape[:2]

        # Математика наклонных вертикальных линий
        angle_rad = math.radians(angle_deg)
        tan_a = math.tan(angle_rad)

        center_x = int(width * pos_x_coeff)
        half_height = height / 2
        half_w_thick = (width * width_coeff) / 2

        def get_line_left_x(y: float) -> int:
            return int(center_x - half_w_thick + (y - half_height) * tan_a)

        def get_line_right_x(y: float) -> int:
            return int(center_x + half_w_thick + (y - half_height) * tan_a)

        # Рисуем полигон зоны детекции
        pts = np.array(
            [
                [get_line_left_x(0), 0],
                [get_line_right_x(0), 0],
                [get_line_right_x(float(height)), height],
                [get_line_left_x(float(height)), height],
            ],
            np.int32,
        )

        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], (0, 255, 255))
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

        # Трекинг пешеходов (conf=0.30 достаточно для уверенного обнаружения)
        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            conf=0.3,
            verbose=False,
        )

        if results and len(results) > 0:
            boxes_object = results[0].boxes
            if boxes_object is not None and boxes_object.id is not None:
                boxes = boxes_object.xyxy.tolist()
                track_ids = boxes_object.id.int().tolist()

                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = box

                    # Из-за машин берем геометрический центр туловища (грудь/пояс)
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    # Визуализация трека и ID человека на экране
                    cv2.rectangle(
                        frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2
                    )
                    cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
                    cv2.putText(
                        frame,
                        f"ID: {track_id}",
                        (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )

                    # Границы коридора для текущей высоты объекта
                    limit_left = get_line_left_x(float(cy))
                    limit_right = get_line_right_x(float(cy))

                    # Логика трекинга состояний (пересечение всей зоны)
                    if track_id not in track_states:
                        if cx < limit_left:
                            track_states[track_id] = "LEFT"
                        elif cx > limit_right:
                            track_states[track_id] = "RIGHT"
                    else:
                        state = track_states[track_id]
                        if state == "LEFT" and cx > limit_right:
                            log_pedestrian(track_id=track_id, direction="OUT")
                            track_states[track_id] = "COUNTED"
                            counted_ids.add(track_id)
                            print(f"[DATABASE] Pedestrian {track_id} moved RIGHT (OUT)")
                        elif state == "RIGHT" and cx < limit_left:
                            log_pedestrian(track_id=track_id, direction="IN")
                            track_states[track_id] = "COUNTED"
                            counted_ids.add(track_id)
                            print(f"[DATABASE] Pedestrian {track_id} moved LEFT (IN)")

                    track_history[track_id] = (cx, cy)

        # Рисуем ограничительные линии
        cv2.line(
            frame,
            (get_line_left_x(0), 0),
            (get_line_left_x(float(height)), height),
            (0, 165, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.line(
            frame,
            (get_line_right_x(0), 0),
            (get_line_right_x(float(height)), height),
            (0, 165, 255),
            2,
            cv2.LINE_AA,
        )

        # Пишем строго НА ЛАТИНИЦЕ во избежание багов отображения
        cv2.putText(
            frame,
            f"Total Count: {len(counted_ids)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Shawarma Horizontal Traffic Analyzer", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_analytics("data/IMG_1686.MOV")
