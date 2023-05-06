import time
import cv2
import mediapipe as mp

def main():
  model_path = './models/deeplabv3.tflite'

  BaseOptions = mp.tasks.BaseOptions
  ImageSegmenter = mp.tasks.vision.ImageSegmenter
  ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
  VisionRunningMode = mp.tasks.vision.RunningMode

  # Create a image segmenter instance with the image mode:
  options = ImageSegmenterOptions(
      base_options=BaseOptions(model_asset_path=model_path),
      running_mode=VisionRunningMode.VIDEO,
      output_type=ImageSegmenterOptions.OutputType.CATEGORY_MASK)

  cap = cv2.VideoCapture(0)

  with ImageSegmenter.create_from_options(options) as segmenter:
    while cap.isOpened():
      # そのフレームの画像を読み込む
      success, frame = cap.read()
      if success is False:
        print("Ignoring empty camera frame.")
        break

      # 画像データをmediapipe用に変換する
      mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
      segmented_masks = segmenter.segment_for_video(mp_image, int(time.time() * 1000))

      mask = segmented_masks[0].numpy_view()
      cv2.imshow('mask', cv2.applyColorMap(mask, cv2.COLORMAP_JET))
      key = cv2.waitKey(1)&0xFF
      if key == ord('q'):
        break

  cap.release()

if __name__=='__main__':
  main()