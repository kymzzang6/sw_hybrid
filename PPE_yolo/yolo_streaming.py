from ultralytics import YOLO
import cv2
import time

# ====== 사용자 설정 ======
model_path = 'PPE_yolo/yolo_weights/ft12_v11_2_best/best.pt'  # .pt 파일 경로 (파인튜닝한 모델 경로로 수정)
webcam_index = 0           # 웹캠 인덱스 (0, 1, 2, ...)

# ====== 모델 로드 ======
try:
    model = YOLO(model_path)
except Exception as e:
    print(f"모델 로드 중 오류 발생: {e}")
    exit()

# ====== 웹캠 열기 ======
cap = cv2.VideoCapture(webcam_index)
if not cap.isOpened():
    print(f"웹캠(인덱스: {webcam_index})을 열 수 없습니다.")
    exit()

print("웹캠 객체 탐지를 시작합니다. 종료하려면 'q' 키를 누르세요.")

prev_time = 0

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        print("웹캠에서 프레임을 읽을 수 없습니다.")
        break

    # ====== 실시간 추론 ======
    # model.predict() 또는 model()을 사용, stream=True로 메모리 효율적 처리 가능
    results = model(frame, stream=False) # stream=True 설정 시 결과는 generator 형태

    # FPS 계산
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # 결과 프레임 가져오기 (결과가 자동으로 그려짐)
    annotated_frame = results[0].plot()

    # FPS 정보 추가
    cv2.putText(annotated_frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 화면에 출력
    cv2.imshow("YOLOv11 Real-Time Detection", annotated_frame)

    # 'q' 키를 누르면 루프 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ====== 자원 해제 ======
cap.release()
cv2.destroyAllWindows()
print("객체 탐지를 종료합니다.")
