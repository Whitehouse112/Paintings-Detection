import numpy as np
import cv2
from painting_detection import detect_paintings, init_histogram
from painting_rectification import rectify_paintings, init_rectification
from painting_retrieval import retrieve_paintings, init_database
from utility import draw, load_video, plot_f_histogram


video_name = 'GOPR5826.MP4'
video = load_video(video_name)

print("Initializing histogram...")
init_histogram()
init_rectification()
print("Initializing ORB database...")
init_database()
print("Done")
f_list = []

while video.grab():
    _, frame = video.retrieve()
    frame = np.swapaxes(np.swapaxes(frame, 0, 2), 1, 2)  # (3, H, W)
    print("\nFrame", int(video.get(cv2.CAP_PROP_POS_FRAMES)) - 1)

    roi_list, cont_list = detect_paintings(np.array(frame))
    paintings, f_list = rectify_paintings(cont_list, np.array(frame))
    retrieved = retrieve_paintings(paintings)
    
    # Show results
    print("ROI list:", roi_list)
    draw(roi_list, paintings, retrieved, np.array(frame))

    # Delay & escape-key
    # video.set(cv2.CAP_PROP_POS_FRAMES, int(video.get(cv2.CAP_PROP_POS_FRAMES)) + int(video.get(cv2.CAP_PROP_FPS)))
    if cv2.waitKey(1) == ord('q'):  # pausa
        plot_f_histogram(f_list)
        if cv2.waitKey() == ord('q'):  # esci
            break
        else:  # continua
            continue
    # cv2.waitKey()

plot_f_histogram(f_list)
video.release()
cv2.destroyAllWindows()
