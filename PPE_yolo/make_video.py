import cv2
import os
from datetime import datetime

# 저장할 디렉토리 생성
save_dir = 'PPE_yolo/dataset/custom_dataset/custom'
os.makedirs(save_dir, exist_ok=True)

# 웹캠 열기
cap = cv2.VideoCapture(2)

# 해상도 설정 (선택사항)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_count = 0
save_interval = 10  # 몇 프레임마다 저장할지

print("프레임 캡처 시작. 'q'를 누르면 종료, 's'를 누르면 즉시 저장합니다.")

while cap.isOpened():
    ret, frame = cap.read()
    
    if ret:
        frame_count += 1
        
        # 화면에 표시
        cv2.imshow('Frame Capture', frame)
        
        # 일정 간격마다 자동 저장
        if frame_count % save_interval == 0:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            filename = f'{save_dir}/frame_{timestamp}.jpg'
            cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f'저장됨: {filename}')
        
        # 키 입력 처리
        key = cv2.waitKey(1) & 0xFF
        
        # 's' 키: 수동 저장
        if key == ord('s'):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            filename = f'{save_dir}/frame_manual_{timestamp}.jpg'
            cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f'수동 저장됨: {filename}')
        
        # 'q' 키: 종료
        elif key == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
print(f'총 {frame_count}프레임 처리 완료')
