import cv2
import torch
from ultralytics import YOLO
from config import MODEL_PATH
from database import log_pedestrian, init_db, get_line_position


def run_analytics(video_source: str | int = 0):
    """
    Запуск аналитики трафика.
    video_source: путь к файлу mp4 или 0 для веб-камеры / смартфона по USB
    """
    # Инициализируем БД, если её еще нет
    init_db()

    # Проверяем доступность GPU (CUDA/MPS) для ускорения, иначе CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Используем устройство для инференса: {device}")

    # Загружаем модель (она автоматически скачается в weight при первом запуске)
    model = YOLO(MODEL_PATH).to(device)

    # Открываем источник видео
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Ошибка: Не удалось открыть источник видео {video_source}")
        return

    # Словари для отслеживания истории перемещения (чтобы понять направление)
    track_history = {}
    # Множество ID, которые мы уже посчитали (чтобы не дублировать)
    counted_ids = set()

    print("Начался анализ видеопотока. Нажмите 'q' для выхода.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Видеопоток завершен или потеряно соединение.")
            break

        # Динамически запрашиваем коэффициент из БД (измененный в Streamlit)
        dynamic_coeff = get_line_position()

        # Берем реальное разрешение текущего кадра
        height, width = frame.shape[:2]
        line_y = int(height * dynamic_coeff)

        # Возвращаем запуск трекера YOLOv8, который потерялся при копировании
        results = model.track(
            frame, persist=True, tracker="bytetrack.yaml", classes=[0], verbose=False
        )

        # Безопасно извлекаем объект boxes, проверяя его существование
        if results and len(results) > 0:
            boxes_object = results[0].boxes

            # Явная проверка для Pylance, что объект boxes и его id не равны None
            if boxes_object is not None and boxes_object.id is not None:
                # Используем # type: ignore, чтобы линтер не ругался на динамические методы Ultralytics
                boxes = boxes_object.xyxy.cpu().numpy()  # type: ignore
                track_ids = boxes_object.id.cpu().numpy().astype(int)  # type: ignore

                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = box

                    # Точка для детекции — «низ» объекта (ноги пешехода)
                    cx = int((x1 + x2) / 2)
                    cy = int(y2)

                    # Рисуем рамку вокруг человека и его ID
                    cv2.rectangle(
                        frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2
                    )
                    cv2.putText(
                        frame,
                        f"ID: {track_id}",
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0),
                        2,
                    )
                    cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

                    # Логика определения направления при пересечении линии
                    if track_id in track_history:
                        prev_cy = track_history[track_id]

                        # Вариант 1: Человек шел СВЕРХУ ВНИЗ и пересек линию
                        if prev_cy < line_y <= cy and track_id not in counted_ids:
                            counted_ids.add(track_id)
                            log_pedestrian(track_id=track_id, direction="IN")
                            print(
                                f"[БД] Человек {track_id} зафиксирован: Идет К ТОЧКЕ (IN)"
                            )

                        # Вариант 2: Человек шел СНИЗУ ВВЕРХ и пересек линию
                        elif prev_cy > line_y >= cy and track_id not in counted_ids:
                            counted_ids.add(track_id)
                            log_pedestrian(track_id=track_id, direction="OUT")
                            print(
                                f"[БД] Человек {track_id} зафиксирован: Идет ОТ ТОЧКИ (OUT)"
                            )

                    # Обновляем историю позиции для этого ID
                    track_history[track_id] = cy

        # Рисуем виртуальную линию (сделали чуть аккуратнее — толщина 2, сглаживание LINE_AA)
        cv2.line(frame, (0, line_y), (width, line_y), (0, 0, 255), 2)
        cv2.putText(
            frame,
            "Liniya podscheta",
            (50, line_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

        # Выводим текущее количество уникальных прохожих на экран
        cv2.putText(
            frame,
            f"Vsego v baze: {len(counted_ids)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        # Показываем кадр
        cv2.imshow("Shawarma Traffic Analyzer", frame)

        # Выход по нажатию клавиши 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Если на камере Gnome/Wayland зависает, замените 0 на "data/test.mp4"
    run_analytics(0)
