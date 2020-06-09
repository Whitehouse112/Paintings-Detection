# ------------------------------------------------------------------------
# Painting detection project
# By Lorenzo Cuoghi, Federico Panzani, Lorenzo Rosini
# (Computer) Vision and Cognitive Systems 2020
# ------------------------------------------------------------------------
import cv2
import painting_detection as detect
import painting_rectification as rect
import painting_retrieval as retr
import people_detection as people
import utility as util
import time


# ------------------------------------------------------------------------
# MAIN process
# ------------------------------------------------------------------------
def main():
    video_name = "GOPR5826.MP4"
    video = util.init(video_name)

    while video.grab():
        _, frame = video.retrieve()  # (H, W, 3)

        n_frame = int(video.get(cv2.CAP_PROP_POS_FRAMES)) - 1

        time_now = time.time()
        roi_list, cont_list = detect.detect_paintings(frame)
        rectified, roi_list, cont_list = rect.rectify_paintings(roi_list, cont_list, frame)
        room, retrieved = retr.retrieve_paintings(rectified)
        people_boxes = people.detect_people(frame, roi_list)

        print("Computational time =", time.time() - time_now)

        print("\n-----------------------------------")
        print("\nFrame", n_frame)
        print("\nROI list:", roi_list)
        util.print_ranking(retrieved)
        util.print_room(room)
        util.draw(roi_list, cont_list, rectified, retrieved, people_boxes, room, frame)

        # video = util.skip_frames(video, fps=2)
        # Delay & escape-key
        if cv2.waitKey(2) == ord('q'):
            break
        # cv2.waitKey()

    video.release()
    cv2.destroyAllWindows()


# ------------------------------------------------------------------------
# STARTING POINT
# ------------------------------------------------------------------------
if __name__ == "__main__":
    main()

# ------------------------------------------------------------------------
