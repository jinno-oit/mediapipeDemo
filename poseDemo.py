import cv2
from MediapipeClass import MediapipePose

def main():
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

if __name__ == '__main__':
    main()