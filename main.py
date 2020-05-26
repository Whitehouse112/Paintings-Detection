import numpy as np
import cv2
from painting_detection import detect_paintings, init_histogram
from painting_rectification import rectify_paintings, init_rectification
from painting_retrieval import retrieve_paintings, init_database, read_file
from utility import draw, load_video  # , skip_frames


video_name = 'GOPR5826.MP4'
video = load_video(video_name)

print("Initializing histogram...")
init_histogram()
init_rectification()
print("Initializing ORB database...")
init_database()
print("Reading Data from csv file...")
read_file()
print("Done")

while video.grab():
    _, frame = video.retrieve()  # (H, W, 3)
    print("\nFrame", int(video.get(cv2.CAP_PROP_POS_FRAMES)) - 1)

    roi_list, cont_list = detect_paintings(np.array(frame))
    paintings = rectify_paintings(cont_list, np.array(frame))
    room, retrieved = retrieve_paintings(paintings)
    
    # Show results
    print("\nROI list:", roi_list)
    print("\nRoom:", room)
    draw(roi_list, cont_list, paintings, retrieved, np.array(frame))
    print("\n-----------------------------------")

    # Delay & escape-key
    # video = skip_frames(video, fps=1)
    if cv2.waitKey(1) == ord('q'):  # pausa
        if cv2.waitKey() == ord('q'):  # esci
            break
        else:  # continua
            continue
    # cv2.waitKey()

video.release()
cv2.destroyAllWindows()
