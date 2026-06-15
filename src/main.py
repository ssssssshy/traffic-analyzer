import cv2
import torch
import numpy as np
import math
from ultralytics import YOLO
from database import log_pedestrian, init_db, get_setting


def run_analytics(video_source: str | int = "IMG_1686.MOV"):
    init_db()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Загружаем модель. Для мелких объектов на расстоянии рекомендуется использовать yolov8m.pt или yolov8l.pt
    model = YOLO("yolov8n.pt").to(device)

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Не удалось открыть видео: {video_source}")
        return

    track_history = {}
    counted_ids = set()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Считываем настройки горизонтального транзита
        pos_x_coeff = get_setting("line_position_x", 0.5)
        width_coeff = get_setting("line_width", 0.15)
        angle_deg = get_setting("line_angle_v", 0.0)

        height, width = frame.shape[:2]

        # Математика наклонных вертикальных линий: x = y * tan(alpha) + b
        angle_rad = math.radians(angle_deg)
        tan_a = math.tan(angle_rad)

        center_x = int(width * pos_x_coeff)
        half_height = height / 2
        half_w_thick = (width * width_coeff) / 2

        # Функции поиска координаты X для левой и правой границ коридора в зависимости от Y
        def get_line_left_x(y):
            return int(center_x - half_w_thick + (y - half_height) * tan_a)

        def get_line_right_x(y):
            return int(center_x + half_w_thick + (y - half_height) * tan_a)

        # Рисуем вертикальный полигон зоны детекции
        pts = np.array(
            [
                [get_line_left_x(0), 0],
                [get_line_right_x(0), 0],
                [get_line_right_x(height), height],
                [get_line_left_x(height), height],
            ],
            np.int32,
        )

        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], (0, 255, 255))
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

        # Трекинг пешеходов (класс 0). Увеличиваем conf для исключения ложных детекций в статичных объектах
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
                boxes = boxes_object.xyxy.cpu().numpy()  # type: ignore
                track_ids = boxes_object.id.cpu().numpy().astype(int)  # type: ignore

                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = box

                    # ВАЖНО: Из-за припаркованых машин берем центр BBox (грудь/пояс), а не ноги
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    # Отрисовка трека
                    cv2.rectangle(
                        frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2
                    )
                    cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

                    # Вычисляем динамические границы коридора строго на высоте центра тела пешехода (cy)
                    limit_left = get_line_left_x(cy)
                    limit_right = get_line_right_x(cy)

                    if track_id in track_history:
                        prev_cx, prev_cy = track_history[track_id]
                        prev_limit_left = get_line_left_x(prev_cy)
                        prev_limit_right = get_line_right_x(prev_cy)

                        # Движение СЛЕВА НАПРАВО (Вошел левее левой, вышел правее правой)
                        if (
                            prev_cx <= prev_limit_left
                            and cx >= limit_right
                            and track_id not in counted_ids
                        ):
                            counted_ids.add(track_id)
                            log_pedestrian(
                                track_id=track_id, direction="OUT"
                            )  # Движение направо
                            print(f"[БД] Пешеход {track_id} прошел направо (OUT)")

                        # Движение СПРАВА НАЛЕВО (Вошел правее правой, вышел левее левой)
                        elif (
                            prev_cx >= prev_limit_right
                            and cx <= limit_left
                            and track_id not in counted_ids
                        ):
                            counted_ids.add(track_id)
                            log_pedestrian(
                                track_id=track_id, direction="IN"
                            )  # Движение налево
                            print(f"[БД] Пешеход {track_id} прошел налево (IN)")

                    track_history[track_id] = (cx, cy)

        # Отрисовка линий коридора
        cv2.line(
            frame,
            (get_line_left_x(0), 0),
            (get_line_left_x(height), height),
            (0, 165, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.line(
            frame,
            (get_line_right_x(0), 0),
            (get_line_right_x(height), height),
            (0, 165, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f"Посчитано транзитов: {len(counted_ids)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
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
    run_analytics("/data/IMG_1686.mov")
