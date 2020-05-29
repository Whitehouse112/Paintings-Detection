import numpy as np
import cv2
import painting_detection as detect
import painting_rectification as rect
import painting_retrieval as retr
import people_detection as people
import utility as util


video_name = 'VIRB0399.MP4'
video = util.load_video(video_name)

print('\n')
print("Initializing histogram...")
detect.init_histogram()
rect.init_rectification()
print("Initializing ORB database...")
retr.init_database()
print("Reading data from csv file...")
retr.read_file()
print("Done")

# video.set(cv2.CAP_PROP_POS_FRAMES, 1470)
while video.grab():
    _, frame = video.retrieve()  # (H, W, 3)
    print("\n-----------------------------------")
    print("\nFrame", int(video.get(cv2.CAP_PROP_POS_FRAMES)) - 1)

    roi_list, cont_list = detect.detect_paintings(frame)
    rectified = rect.rectify_paintings(cont_list, frame)
    room, retrieved = retr.retrieve_paintings(rectified)
    people_boxes = people.detect_people(np.array(frame))
    
    # Show results
    print("\nROI list:", roi_list)
    util.print_ranking(retrieved)
    util.print_room(room)
    util.draw(roi_list, cont_list, rectified, retrieved, people_boxes, room, frame)

    # Delay & escape-key
    # video = skip_frames(video, fps=1)
    if cv2.waitKey(2) == ord('q'):
        break
    # cv2.waitKey()

video.release()
cv2.destroyAllWindows()
