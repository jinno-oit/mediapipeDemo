# coding: utf-8
import numpy as np
import cv2
import mediapipe as mp


class MediapipeClass():
    def __init__(self):
        """Mediapipe描画用準備"""
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def changeXYZ(self, landmark, width, height):
        landmark_xyz = []
        for point in landmark:
            landmark_xyz.append(np.array([point.x*width, point.y*height, point.z*width]))
        return landmark_xyz


class MediapipeHands(MediapipeClass):
    def __init__(self, max_num_hands=2, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        super().__init__()
        """Hands準備"""
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence)

    def drawHand(self, img, hand):
        if hand.hand_landmarks:
            self.mp_drawing.draw_landmarks(
                img,
                hand.hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style())

    def drawHandAll(self, img, hands):
        self.drawHand(img, hands.Right)
        self.drawHand(img, hands.Left)

    def detectHands(self, img_input):
        img = img_input.copy()
        height, width = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb.copy())

        self.hands_data = HandsData(results.multi_hand_landmarks)
        if results.multi_hand_landmarks:
            # for hand_landmarks, hand_world_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_hand_world_landmarks, results.multi_handedness):
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # https://google.github.io/mediapipe/images/mobile/hand_landmarks.png
                if handedness.classification[0].label == 'Left':
                    self.hands_data.Left.hand_landmarks = hand_landmarks
                    self.hands_data.Left.landmark = self.changeXYZ(hand_landmarks.landmark, width, height)
                if handedness.classification[0].label == 'Right':
                    self.hands_data.Right.hand_landmarks = hand_landmarks
                    self.hands_data.Right.landmark = self.changeXYZ(hand_landmarks.landmark, width, height)
        return self.hands_data


class MediapipePose(MediapipeClass):
    def __init__(self, model_complexity=1, enable_segmentation=True, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        super().__init__()
        """Pose準備"""
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            model_complexity=model_complexity,
            enable_segmentation=enable_segmentation,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence)
        self.is_segmentation = enable_segmentation

    def drawPose(self, img, pose):
        if pose.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                img,
                pose.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())

    def getVisibility(self, landmark):
        visivility = []
        for point in landmark:
            visivility.append(point.visibility)
        return visivility

    def detectPose(self, img_input):
        img = img_input.copy()
        height, width = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb.copy())

        self.pose_data = PoseData()
        if results.pose_landmarks:
            # https://google.github.io/mediapipe/images/mobile/pose_tracking_full_body_landmarks.png
            self.pose_data.pose_landmarks = results.pose_landmarks
            self.pose_data.landmark = self.changeXYZ(results.pose_landmarks.landmark, width, height)
            self.pose_data.visibility = self.getVisibility(results.pose_landmarks.landmark)
            if self.is_segmentation:
                self.pose_data.segmentation_mask = results.segmentation_mask
        return self.pose_data


class MediapipeFaceMesh(MediapipeClass):
    def __init__(self, max_num_faces=1, refine_landmarks=False, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        super().__init__()
        """FaceMesh準備"""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces = max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence)

    def drawFaceMesh(self, img, face_landmarks, is_tesselation=True, is_contours=True, is_irises=True):
        if is_tesselation:
            self.mp_drawing.draw_landmarks( # meshの表示
                image=img,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None, # 座標表示なし
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style())
        if is_contours:
            self.mp_drawing.draw_landmarks( # 輪郭の表示
                image=img,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None, # 座標表示なし
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style())
        # if is_irises:
        #     self.mp_drawing.draw_landmarks( # 目の表示
        #         image=img,
        #         landmark_list=face_landmarks,
        #         connections=self.mp_face_mesh.FACEMESH_IRISES,
        #         landmark_drawing_spec=None, # 座標表示なし
        #         connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_iris_connections_style())

    def drawFaceMeshAll(self, img, face_mesh, is_tesselation=True, is_contours=True, is_irises=True):
        # print(len(face_mesh.face_landmarks))
        for i in range(len(face_mesh.face_landmarks)): # 全ての顔に対して，描画処理を適用
            self.drawFaceMesh(img, face_mesh.face_landmarks[i], is_tesselation=is_tesselation, is_contours=is_contours, is_irises=is_irises)

    def detectFaceMesh(self, img_input):
        img = img_input.copy()
        height, width = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb.copy())

        self.face_mesh_data = FaceMeshData(results.multi_face_landmarks)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
                self.face_mesh_data.face_landmarks.append(face_landmarks)
                self.face_mesh_data.landmark.append(self.changeXYZ(face_landmarks.landmark, width, height))
        return self.face_mesh_data

#############################################################

class Point3D():
    def calcAngle(self, vec1, vec2):
        vec1_length = np.linalg.norm(vec1)
        vec2_length = np.linalg.norm(vec2)
        cos_theta = np.inner(vec1, vec2) / (vec1_length * vec2_length)
        return np.rad2deg(np.arccos(cos_theta))
    
    def calcDist(self, point1, point2):
        # sum = 0
        # for p1, p2 in zip(point1, point2):
        #     sum += (p1 - p2) ** 2
        # return np.sqrt(sum)
        return np.linalg.norm(point1 - point2)

# Hands Data Class ###########################################
class HandsData():
    def __init__(self, mluti_hand_landmarks):
        self.multi_hand_landmarks = mluti_hand_landmarks
        self.Left = HandData('Left')
        self.Right = HandData('Right')
    

class HandData(Point3D):
    def __init__(self, handedness):
        self.handedness = handedness
        self.hand_landmarks = None
        self.landmark = None

    def judgeOpen(self, finger_id):
        if self.landmark is None:
            return None
        th_angle = 140
        if finger_id == 'thumb':
            ind = [1, 2, 3]
            th_angle = 155
        elif finger_id == 'index':
            ind = [5, 6, 7]
        elif finger_id == 'middle':
            ind = [9, 10, 11]
        elif finger_id == 'ring':
            ind = [13, 14, 15]
        elif finger_id == 'pinky':
            ind = [17, 18, 19]
        vec1 = self.landmark[ind[0]] - self.landmark[ind[1]]
        vec2 = self.landmark[ind[2]] - self.landmark[ind[1]]
        if self.calcAngle(vec1, vec2) > th_angle:
            # print(self.handedness+': '+finger_id+' finger is opened.')
            return True
        else:
            # print(self.handedness+': '+finger_id+' finger is bended.')
            return False

    def judgeAll(self):
        if self.landmark is None:
            return
        thumb = 'o' if self.judgeOpen('thumb') else 'x'
        index = 'o' if self.judgeOpen('index') else 'x'
        middle = 'o' if self.judgeOpen('middle') else 'x'
        ring = 'o' if self.judgeOpen('ring') else 'x'
        pinky = 'o' if self.judgeOpen('pinky') else 'x'
        print(thumb, '|', index, middle, ring, pinky)

    def getPoint(self, point_id):
        if self.landmark is None:
            return
        return self.landmark[point_id]


class PoseData(Point3D):
    def __init__(self):
        self.pose_landmarks = None
        self.landmark = None
        self.visibility = None
        self.segmentation_mask = None

    def getPoint(self, point_id):
        if self.landmark is None:
            return
        return self.landmark[point_id]


class FaceMeshData(Point3D):
    def __init__(self, mluti_face_landmarks):
        self.multi_face_landmarks = mluti_face_landmarks
        self.face_landmarks = []
        self.landmark = []

    def getPoint(self, face_id, point_id):
        if len(self.landmark) == 0:
            return
        return self.landmark[face_id][point_id]


# test ###############################################################

def testHands():
    cap = cv2.VideoCapture(0)
    mph = MediapipeHands()
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is None:
            print('cannot read frame.')

        frame = cv2.flip(frame, 1)
        hands = mph.detectHands(frame)

        # print(hands.Left.getPoint(8))
        # hands.Left.judgeOpen('thumb')
        # hands.Right.judgeAll()

        mph.drawHandAll(frame, hands)

        cv2.imshow('res', frame)
        key = cv2.waitKey(1)&0xFF

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyWindow('res')

def testPose():
    cap = cv2.VideoCapture(0)
    mpp = MediapipePose()
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is None:
            print('cannot read frame.')

        frame = cv2.flip(frame, 1)
        pose = mpp.detectPose(frame)

        # print(np.array(pose.visibility) > 0.5)
        # cv2.imshow('mask', pose.segmentation_mask)

        mpp.drawPose(frame, pose)

        cv2.imshow('res', frame)
        key = cv2.waitKey(1)&0xFF

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyWindow('res')

def testFaceMesh():
    cap = cv2.VideoCapture(0)
    mpf = MediapipeFaceMesh()
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is None:
            print('cannot read frame.')

        frame = cv2.flip(frame, 1)
        face_mesh = mpf.detectFaceMesh(frame)

        # print(face_mesh.getPoint(0, 0))

        # if len(face_mesh.face_landmarks) > 0:
        #     mpf.drawFaceMesh(frame, face_mesh.face_landmarks[0])

        mpf.drawFaceMeshAll(frame, face_mesh)

        cv2.imshow('res', frame)
        key = cv2.waitKey(1)&0xFF

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyWindow('res')

if __name__=='__main__':
    testHands()
    testPose()
    testFaceMesh()
