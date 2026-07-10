import cv2
import mediapipe as mp
import time

mpHands = mp.solutions.hands
hands = mpHands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mpDraw = mp.solutions.drawing_utils

tipIds = [4, 8, 12, 16, 20]

cap = cv2.VideoCapture(0)

pTime = 0

while True:

    success, img = cap.read()

    if not success:
        break

    img = cv2.flip(img, 1)

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(imgRGB)

    totalFingers = 0

    if results.multi_hand_landmarks:

        for handNo, handLms in enumerate(results.multi_hand_landmarks):

            mpDraw.draw_landmarks(
                img,
                handLms,
                mpHands.HAND_CONNECTIONS
            )

            lmList = []

            h, w, c = img.shape

            for id, lm in enumerate(handLms.landmark):

                cx = int(lm.x * w)
                cy = int(lm.y * h)

                lmList.append((cx, cy))

            if len(lmList) != 0:

                fingers = []

                handType = results.multi_handedness[handNo].classification[0].label

                # Thumb

                if handType == "Right":

                    if lmList[tipIds[0]][0] < lmList[tipIds[0]-1][0]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                else:

                    if lmList[tipIds[0]][0] > lmList[tipIds[0]-1][0]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                # Other fingers

                for id in range(1,5):

                    if lmList[tipIds[id]][1] < lmList[tipIds[id]-2][1]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                totalFingers = fingers.count(1)

                cv2.putText(
                    img,
                    f"{handType} Hand",
                    (10,40+handNo*40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,255,255),
                    2
                )

    cv2.rectangle(img,(10,70),(230,170),(0,255,0),-1)

    cv2.putText(
        img,
        f"Fingers: {totalFingers}",
        (25,135),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255,255,255),
        3
    )

    cTime = time.time()

    fps = 1/(cTime-pTime)

    pTime = cTime

    cv2.putText(
        img,
        f"FPS: {int(fps)}",
        (10,220),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255,0,255),
        2
    )

    cv2.imshow("Hand Gesture Recognition", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()