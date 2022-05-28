import cv2
from MediapipeClass import MediapipeHands

def main():
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

if __name__ == '__main__':
    main()