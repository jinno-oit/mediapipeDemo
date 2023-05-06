# https://developers.google.com/mediapipe/solutions/vision/hand_landmarker

import time
import cv2
import mediapipe as mp
import download as dl

MARGIN_X = 10
MARGIN_Y = 30
RIGHT_HAND_COLOR = (0, 255, 0)
LEFT_HAND_COLOR = (100, 100, 255)

def visualize_hand(img, hand_landmarker_result):
  h, w = img.shape[:2]
  for hand, info in zip(hand_landmarker_result.hand_landmarks, hand_landmarker_result.handedness):
    if info[0].category_name == 'Right':
      color = RIGHT_HAND_COLOR
    else:
      color = LEFT_HAND_COLOR
    for point in hand:
      cv2.circle(img, (int(point.x*w), int(point.y*h)), 5, color, thickness=2)
    txt = info[0].category_name+'('+'{:#.2f}'.format(info[0].score)+')'
    wrist_point_for_text = (int(hand[0].x*w)+MARGIN_X, int(hand[0].y*h)+MARGIN_Y)
    cv2.putText(img, org=wrist_point_for_text, text=txt, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=color, thickness=1, lineType=cv2.LINE_4)
  return img

def main():
  base_url = 'https://storage.googleapis.com/mediapipe-tasks/hand_landmarker/'
  model_folder_path = './models'
  model_name = 'hand_landmarker.task'
  model_path = dl.set_model(base_url, model_folder_path, model_name)

  # 初期設定（名前が長いのでリネームしている）
  BaseOptions = mp.tasks.BaseOptions
  HandLandmarker = mp.tasks.vision.HandLandmarker
  HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
  VisionRunningMode = mp.tasks.vision.RunningMode

  # HandLandmarkerのオブション
  options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    num_hands=2,
    running_mode=VisionRunningMode.VIDEO)

  cap = cv2.VideoCapture(0)

  # detectorをオープンして処理を開始する
  with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
      # そのフレームの画像を読み込む
      success, frame = cap.read()
      if success is False:
        print("Ignoring empty camera frame.")
        break

      fliped_frame = cv2.flip(frame, 1) # 左右反転

      # 画像データをmediapipe用に変換する
      mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=fliped_frame)

      # オブジェクト検出を実行する
      hand_landmarker_result = landmarker.detect_for_video(mp_image, int(time.time() * 1000))

      # 検出したオブジェクトを囲う枠，および名前とスコアを画像に重畳する
      annotated_image = visualize_hand(fliped_frame.copy(), hand_landmarker_result)

      # 画像を表示する（'q'キーを押すとループ終了）
      cv2.imshow('result', annotated_image)
      key = cv2.waitKey(1)&0xFF
      if key == ord('q'):
        break

  cap.release()

if __name__=='__main__':
  main()

