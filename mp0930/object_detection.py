import mediapipe as mp
import cv2
import time
import download as dl

def visualize(img, detection_result, threshold):
  for obj in detection_result.detections:
    if obj.categories[0].score > threshold:
      # 枠の左上座標，右下座標を算出し，描画する
      upper_left_point = (obj.bounding_box.origin_x, obj.bounding_box.origin_y)
      lower_right_point = (obj.bounding_box.origin_x+obj.bounding_box.width, obj.bounding_box.origin_y+obj.bounding_box.height)
      cv2.rectangle(img, upper_left_point, lower_right_point, (0,0,255), thickness=1)
      # 枠の左上あたりにカテゴリ名を表示する
      txt = obj.categories[0].category_name+'('+'{:#.2f}'.format(obj.categories[0].score)+')'
      lower_left_point_for_text = (obj.bounding_box.origin_x+10, obj.bounding_box.origin_y+30)
      cv2.putText(img, org=lower_left_point_for_text, text=txt, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(0,0,255), thickness=1, lineType=cv2.LINE_4)
  return img

def main():
  # 以下からモデルをダウンロードしてセットする
  # https://developers.google.com/mediapipe/solutions/vision/object_detector/index#models
  base_url = 'https://storage.googleapis.com/mediapipe-tasks/object_detector/'
  model_folder_path = './models'
  model_name = 'efficientdet_lite0_fp32.tflite'
  model_path = dl.set_model(base_url, model_folder_path, model_name)

  # 初期設定（名前が長いのでリネームしている）
  BaseOptions = mp.tasks.BaseOptions
  ObjectDetector = mp.tasks.vision.ObjectDetector
  ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
  VisionRunningMode = mp.tasks.vision.RunningMode

  # ObjectDetectionのオブション
  options = ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    max_results=5,
    running_mode=VisionRunningMode.VIDEO)

  cap = cv2.VideoCapture(0)

  # detectorをオープンして処理を開始する
  with ObjectDetector.create_from_options(options) as detector:
    while cap.isOpened():
      # そのフレームの画像を読み込む
      success, frame = cap.read()
      if success is False:
        print("Ignoring empty camera frame.")
        break

      # 画像データをmediapipe用に変換する
      mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

      # オブジェクト検出を実行する
      detection_result = detector.detect_for_video(mp_image, int(time.time() * 1000))

      # 検出したオブジェクトを囲う枠，および名前とスコアを画像に重畳する
      threshold = 0.5 # 描画するオブジェクトの最低スコア
      annotated_image = visualize(frame.copy(), detection_result, threshold)

      # 画像を表示する（'q'キーを押すとループ終了）
      cv2.imshow('result', annotated_image)
      key = cv2.waitKey(1)&0xFF
      if key == ord('q'):
        break

  cap.release()

if __name__=='__main__':
      main()
