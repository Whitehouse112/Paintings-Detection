import numpy as np
import cv2
from painting_detection import detect_paintings, init_histogram
from painting_rectification import rectify_paintings, init_rectification
from orb import orb_retrieve, init_database
from utility import draw, load_video

video_name = 'VIRB0392.MP4'
video = load_video(video_name)

print("Initializing histogram...")
init_histogram()
init_rectification()
print("Initializing ORB database...")
init_database()
print("Done")

while video.grab():
    _, frame = video.retrieve()
    frame = np.swapaxes(np.swapaxes(frame, 0, 2), 1, 2)  # (3, H, W)
    print("\nFrame", int(video.get(cv2.CAP_PROP_POS_FRAMES)) - 1)

    roi_list, cont_list = detect_paintings(np.array(frame))
    paintings = rectify_paintings(cont_list, np.array(frame))
    
    # Show results
    print("ROI list:", roi_list)
    draw(roi_list, paintings, np.array(frame))

    retrieved = orb_retrieve(paintings)
    if not(retrieved is None):
        cv2.imshow("Retrieved paintings", np.concatenate(retrieved, axis=1))

    # Delay & escape-key
    #video.set(cv2.CAP_PROP_POS_FRAMES, int(video.get(cv2.CAP_PROP_POS_FRAMES)) + int(video.get(cv2.CAP_PROP_FPS)))
    if cv2.waitKey(1) == ord('q'):  # pausa
        if cv2.waitKey() == ord('q'):  # esci
            break
        else:  # continua
            continue
    #cv2.waitKey()

video.release()
cv2.destroyAllWindows()
