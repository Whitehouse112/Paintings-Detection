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


# ------------------------------------------------------------------------
# MAIN process
# ------------------------------------------------------------------------
def main():
    video = util.init("VIRB0392.MP4", start_frame=0, default_fps=2)

    while not util.video_end(video):
        frame = util.get_next_frame(video)  # shape = (H, W, 3), colorspace = BGR

        roi_list, cont_list = detect.detect_paintings(frame)
        rectified, roi_list, cont_list = rect.rectify_paintings(roi_list, cont_list, frame)
        room, retrieved = retr.retrieve_paintings(rectified)
        people_boxes = people.detect_people(frame, roi_list)

        util.show_results(video, frame, roi_list, cont_list, rectified, retrieved, room, people_boxes)

        if cv2.waitKey(1) > 0:  # exit
            break
        # cv2.waitKey()  # one frame at a time

    util.close_all(video)


# ------------------------------------------------------------------------
# STARTING POINT
# ------------------------------------------------------------------------
if __name__ == "__main__":
    main()

# ------------------------------------------------------------------------
