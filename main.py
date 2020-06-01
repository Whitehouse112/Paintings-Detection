import cv2
import painting_detection as detect
import painting_rectification as rect
import painting_retrieval as retr
import people_detection as people
import utility as util
import threading


outputs = {}


def paintingsThread(frame, roi_list, cont_list):
    rectified, roi_list, cont_list = rect.rectify_paintings(roi_list, cont_list, frame)
    room, retrieved = retr.retrieve_paintings(rectified)
    outputs["roi_list"] = roi_list
    outputs["cont_list"] = cont_list 
    outputs["rectified"] = rectified
    outputs["retrieved"] = retrieved
    outputs["room"] = room


def peopleThread(frame, roi_list):
    people_boxes = people.detect_people(frame, roi_list)
    outputs['people_boxes'] = people_boxes


def main():
    video_name = 'VIRB0392.MP4'
    video = util.load_video(video_name)

    print('\n')
    print("Initializing histogram...")
    detect.init_histogram()
    rect.init_rectification()
    print("Initializing ORB database...")
    retr.init_database()
    print("Reading data from CSV file...")
    retr.read_file()
    print("Done")

    while video.grab():
        _, frame = video.retrieve()  # (H, W, 3)
        print("\n-----------------------------------")
        print("\nFrame", int(video.get(cv2.CAP_PROP_POS_FRAMES)) - 1)

        roi_list, cont_list = detect.detect_paintings(frame)

        t1 = threading.Thread(target=paintingsThread, args=(frame, roi_list, cont_list))
        t2 = threading.Thread(target=peopleThread, args=(frame, roi_list))

        t1.start()
        t2.start()

        t1.join()
        t2.join()

        roi_list = outputs["roi_list"]
        cont_list = outputs["cont_list"]
        rectified = outputs["rectified"]
        retrieved = outputs["retrieved"]
        room = outputs["room"]
        people_boxes = outputs['people_boxes']

        # Show results
        print("\nROI list:", roi_list)
        util.print_ranking(retrieved)
        util.print_room(room)
        util.draw(roi_list, cont_list, rectified, retrieved, people_boxes, room, frame)
        # util.draw(roi_list, cont_list, rectified, [], [], 0, frame)

        # Delay & escape-key
        # video = skip_frames(video, fps=1)
        if cv2.waitKey(2) == ord('q'):
            break
        # cv2.waitKey()

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
