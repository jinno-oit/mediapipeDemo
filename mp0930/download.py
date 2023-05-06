import urllib.request
import os

def set_model(base_url, model_folder_path, model_name):
  model_path = model_folder_path+'/'+model_name

  # modelファイルが存在しない場合，ダウンロードしてくる
  if not os.path.exists(model_path):
    # model_folderが存在しない場合，フォルダを作成する
    if not os.path.exists(model_folder_path):
        os.mkdir(model_folder_path)
    # モデルをダウンロードする
    url = base_url+model_name
    save_name = model_path
    urllib.request.urlretrieve(url, save_name)

  return model_path

if __name__=='__main__':

  # object detection
  base_url = 'https://storage.googleapis.com/mediapipe-tasks/object_detector/'
  model_folder_path = './models'
  model_name = 'efficientdet_lite0_fp32.tflite'

  # hand landmarker
  base_url = 'https://storage.googleapis.com/mediapipe-tasks/hand_landmarker/'
  model_folder_path = './models'
  model_name = 'hand_landmarker.task'

  # pose landmarker (unreleased)
  base_url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/'
  model_folder_path = './models'
  model_name = 'pose_landmarker_heavy.task'



  model_path = set_model(base_url, model_folder_path, model_name)

