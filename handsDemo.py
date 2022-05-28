import cv2
from MediapipeClass import MediapipeHands

def main():
    cap = cv2.VideoCapture(0)
    mph = MediapipeHands()
    mode = 1
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is None:
            print('cannot read frame.')

        key = cv2.waitKey(1)&0xFF
        if key >= ord('0') and key <= ord('9'):
            mode = key - ord('0')
        
        frame = cv2.flip(frame, 1)
        hands = mph.detectHands(frame)

        if mode == 1:
            print(hands.Left.getPoint(1, 8))

        elif mode == 2:
            hands.Left.judgeOpen(finger_id='thumb')

        elif mode == 3:
            hands.Right.judgeAll()

        mph.drawHandAll(frame, hands)
        cv2.imshow('res', frame)

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyWindow('res')

if __name__ == '__main__':
    main()
