import cv2
import torch
from ultralytics import YOLO
from config import MODEL_PATH, LINE_POSITION_COEFF
from database import log_pedestrian, init_db


def run_analytics(video_source: str or int = 0):
    """
    Запуск аналитики трафика.
    video_source: путь к файлу mp4 или 0 для веб-камеры / смартфона по USB
    """
    # Инициализируем БД, если её еще нет
    init_db()

    # Проверяем доступность GPU (CUDA/MPS) для ускорения, иначе CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Используем устройство для инференса: {device}")

    # Загружаем модель (она автоматически скачается в корень или weight при первом запуске)
    model = YOLO(MODEL_PATH).to(device)

    # Открываем источник видео
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Ошибка: Не удалось открыть источник видео {video_source}")
        return

    # Получаем размеры кадра для отрисовки линии
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Вычисляем координату Y для нашей виртуальной линии
    line_y = int(height * LINE_POSITION_COEFF)

    # Словари для отслеживания истории перемещения (чтобы понять направление)
    # Ключ: track_id, Значение: предыдущий Y-центр объекта
    track_history = {}
    # Множество ID, которые мы уже посчитали (чтобы не дублировать)
    counted_ids = set()

    print("Начался анализ видеопотока. Нажмите 'q' для выхода.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Видеопоток завершен или потерян соединения.")
            break

        # Запускаем YOLO с трекером ByteTrack
        # classes=[0] означает, что мы детектируем только людей (person)
        results = model.track(
            frame, persist=True, tracker="bytetrack.yaml", classes=[0], verbose=False
        )

        # Проверяем, есть ли обнаруженные объекты с ID трекера
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box

                # Точка для детекции — «низ» объекта (ноги пешехода)
                # Это точнее всего определяет, где человек стоит/идет относительно линии
                cx = int((x1 + x2) / 2)
                cy = int(y2)

                # Рисуем рамку вокруг человека и его ID
                cv2.rectangle(
                    frame, (int(x1), int(y1)), (int(x2), int(int(y2))), (255, 0, 0), 2
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

        # Рисуем виртуальную линию (красная)
        cv2.line(frame, (0, line_y), (width, line_y), (0, 0, 255), 3)
        cv2.putText(
            frame,
            "ЛИНИЯ ПОДСЧЕТА",
            (20, line_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

        # Выводим текущее количество уникальных прохожих на экран
        cv2.putText(
            frame,
            f"Всего в базе: {len(counted_ids)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        # Показываем кадр
        cv2.imshow("Shawarma Traffic Analyzer", frame)

        # Выход по нажатию клавиши 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Для теста подставьте сюда путь к вашему видеофайлу в папке data,
    # например: run_analytics("data/test_street.mp4")
    # Если хотите потестить на веб-камере ноутбука — оставьте 0.
    run_analytics(0)
