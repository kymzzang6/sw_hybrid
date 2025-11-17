from ultralytics import YOLO
import matplotlib as plt 




model = YOLO('yolo_weights/ppe1_11s_best/best.pt')


test_img_dir = 'dataset/custom_dataset/custom'

results = model.predict(
    source=test_img_dir,
    save=True, # 결과 자동 저장
    conf=0.2, # 정확도 25% 넘으면 다 검출
    project='runs/test', # 결과 저장할 경로
    name='labeling_ppe', # 이름은 맘대로 지정
    save_txt = True,
    exist_ok=False # True : 같은 이름의 폴더가 이미 존재하면 덮어쓰기 False : 같은 이름 있으면 뒷번호로 새로 만들기
)
