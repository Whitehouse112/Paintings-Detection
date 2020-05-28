import numpy as np
import cv2
import painting_detection as detect
import painting_rectification as rect
import painting_retrieval as retr
import people_detection as p_detect
import utility as util
import threading
import time

outputs = {}

def paintings(frame):
    roi_list, cont_list = detect.detect_paintings(np.array(frame))
    rectified = rect.rectify_paintings(cont_list, np.array(frame))
    room, retrieved = retr.retrieve_paintings(rectified)
    outputs["roi_list"] = roi_list
    outputs["cont_list"] = cont_list 
    outputs["rectified"] = rectified
    outputs["retrieved"] = retrieved
    outputs["room"] = room


def people(frame):
    frame_people = p_detect.detect_people(np.array(frame))
    outputs['frame_people'] = frame_people


def main():
    video_name = 'GOPR5826.MP4'
    video = util.load_video(video_name)

    print('\n')
    print("Initializing histogram...")
    detect.init_histogram()
    rect.init_rectification()
    print("Initializing ORB database...")
    retr.init_database()
    print("Reading Data from csv file...")
    retr.read_file()
    print("Done")

    while video.grab():
        _, frame = video.retrieve()  # (H, W, 3)
        print("\n-----------------------------------")
        print("\nFrame", int(video.get(cv2.CAP_PROP_POS_FRAMES)) - 1)

        t1 = threading.Thread(target=paintings, args=(frame,))
        t2 = threading.Thread(target=people, args=(frame,))
        
        t1.start()
        t2.start()

        t1.join()
        t2.join()

        roi_list = outputs["roi_list"]
        cont_list = outputs["cont_list"]
        rectified = outputs["rectified"]
        retrieved = outputs["retrieved"]
        room = outputs["room"]
        frame_people = outputs['frame_people']

        # Show results
        print("\nROI list:", roi_list)
        util.print_ranking(retrieved)
        util.print_room(room)
        util.draw(roi_list, cont_list, rectified, retrieved, np.array(frame))

        cv2.namedWindow("Frame people", flags=cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
        cv2.imshow("Frame people", frame_people)

        # Delay & escape-key
        # video = skip_frames(video, fps=1)
        if cv2.waitKey(2) == ord('q'):
            break
        # cv2.waitKey()

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()