import time
import cv2
import mediapipe as mp
import download as dl

def visualize_pose(img, hand_landmarker_result):
  h, w = img.shape[:2]
  # for pose in hand_landmarker_result.pose_landmarks:
  #   color = (0, 255, 0)
  #   for point in hand:
  #     cv2.circle(img, (int(point.x*w), int(point.y*h)), 5, color, thickness=2)
  return img

def main():
  base_url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/'
  model_folder_path = './models'
  model_name = 'pose_landmarker_heavy.task'
  model_path = dl.set_model(base_url, model_folder_path, model_name)

  # 初期設定（名前が長いのでリネームしている）
  BaseOptions = mp.tasks.BaseOptions
  PoseLandmarker = mp.tasks.vision.PoseLandmarker
  PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
  VisionRunningMode = mp.tasks.vision.RunningMode

  # PoseLandmarkerのオブション
  options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO)

  cap = cv2.VideoCapture(0)

  # detectorをオープンして処理を開始する
  with PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
      # そのフレームの画像を読み込む
      success, frame = cap.read()
      if success is False:
        print("Ignoring empty camera frame.")
        break

      # 画像データをmediapipe用に変換する
      mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.flip(frame,1))

      # オブジェクト検出を実行する
      pose_landmarker_result = landmarker.detect_for_video(mp_image, int(time.time() * 1000))

      # 検出したオブジェクトを囲う枠，および名前とスコアを画像に重畳する
      annotated_image = visualize_pose(cv2.flip(frame,1).copy(), pose_landmarker_result)

      # 画像を表示する（'q'キーを押すとループ終了）
      cv2.imshow('result', annotated_image)
      key = cv2.waitKey(1)&0xFF
      if key == ord('q'):
        break

  cap.release()





if __name__=='__main__':
  main()

