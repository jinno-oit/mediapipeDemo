import time
import cv2
import mediapipe as mp
import download as dl

def visualize_face(img, face_detector_result):
  h, w = img.shape[:2]
#   for hand, info in zip(face_detector_result.bounding_box, hand_landmarker_result.handedness):
#     if info[0].category_name == 'Left':
#       color = (0, 255, 0)
#     else:
#       color = (255, 50, 50)
#     for point in hand:
#       cv2.circle(img, (int(point.x*w), int(point.y*h)), 5, color, thickness=2)
#     txt = info[0].category_name+'('+'{:#.2f}'.format(info[0].score)+')'
#     wrist_point_for_text = (int(hand[0].x*w)+10, int(hand[0].y*h)+30)
#     cv2.putText(img, org=wrist_point_for_text, text=txt, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=color, thickness=1, lineType=cv2.LINE_4)
  return img

def main():
  base_url = 'https://storage.googleapis.com/mediapipe-assets/'
  model_folder_path = './models'
  model_name = 'face_detection_short_range.tflite'
  model_path = dl.set_model(base_url, model_folder_path, model_name)

  # 初期設定（名前が長いのでリネームしている）
  BaseOptions = mp.tasks.BaseOptions
  FaceDetector = mp.tasks.vision.FaceDetector
  FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
  VisionRunningMode = mp.tasks.vision.RunningMode

  # FaceDetectorのオブション
  options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO)

  cap = cv2.VideoCapture(0)

  # detectorをオープンして処理を開始する
  with FaceDetector.create_from_options(options) as detector:
    while cap.isOpened():
      # そのフレームの画像を読み込む
      success, frame = cap.read()
      if success is False:
        print("Ignoring empty camera frame.")
        break

      # 画像データをmediapipe用に変換する
      mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

      # オブジェクト検出を実行する
      face_detector_result = detector.detect_for_video(mp_image, int(time.time() * 1000))

      # 検出したオブジェクトを囲う枠，および名前とスコアを画像に重畳する
      annotated_image = visualize_face(frame.copy(), face_detector_result)

      # 画像を表示する（'q'キーを押すとループ終了）
      cv2.imshow('result', annotated_image)
      key = cv2.waitKey(1)&0xFF
      if key == ord('q'):
        break

  cap.release()





if __name__=='__main__':
  main()
