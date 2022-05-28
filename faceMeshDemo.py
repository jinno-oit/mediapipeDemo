import cv2
from MediapipeClass import MediapipeFaceMesh

def main():
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
    main()