import cv2
import painting_detection as detect
import painting_rectification as rect
import painting_retrieval as retr
import people_detection as people
import utility as util
import threading
import queue


processing = None
finished = None
outputs = {}
DetectQueue = queue.Queue()
RectifyQueue = queue.Queue()
RetrieveQueue = queue.Queue()
PeopleQueue = queue.Queue()


class Frame:
    frame = None
    n_frame = None
    roi_list = None
    cont_list = None
    recified = None
    retrieved = None
    room = None
    people_boxes = None

    def __init__(self, frame):
        self.frame = frame


def detectThreadBody(video):
    global DetectQueue, processing, outputs, finished

    while video.grab() and processing:
        _, frame = video.retrieve()  # (H, W, 3)

        while processing and DetectQueue.qsize() > 20:
            pass

        block = Frame(frame)
        block.n_frame = int(video.get(cv2.CAP_PROP_POS_FRAMES)) - 1

        block.roi_list, block.cont_list = detect.detect_paintings(block.frame)

        outputs[block.n_frame] = block
        DetectQueue.put(block.n_frame)
    finished = True


def rectifyThreadBody():
    global DetectQueue, RectifyQueue, outputs, processing

    while processing:
        while processing and RectifyQueue.qsize() > 20:
            pass

        try:
            n_frame = DetectQueue.get(timeout=1)
            block = outputs[n_frame]

            block.rectified, block.roi_list, block.cont_list = rect.rectify_paintings(block.roi_list, block.cont_list,
                                                                                      block.frame)

            outputs[block.n_frame] = block
            RectifyQueue.put(block.n_frame)
        except queue.Empty:
            continue


def retrieveThreadBody():
    global RectifyQueue, RetrieveQueue, outputs, processing

    while processing:
        while processing and RetrieveQueue.qsize() > 20:
            pass

        try:
            n_frame = RectifyQueue.get(timeout=1)
            block = outputs[n_frame]

            block.room, block.retrieved = retr.retrieve_paintings(block.rectified)

            outputs[block.n_frame] = block
            RetrieveQueue.put(block.n_frame)
        except queue.Empty:
            continue


def peopleThreadBody():
    global RetrieveQueue, PeopleQueue, outputs, processing

    while processing:
        while processing and PeopleQueue.qsize() > 20:
            pass

        try:
            n_frame = RetrieveQueue.get(timeout=1)
            block = outputs[n_frame]

            block.people_boxes = people.detect_people(block.frame, block.roi_list)

            outputs[block.n_frame] = block
            PeopleQueue.put(block.n_frame)
        except queue.Empty:
            continue


def main():
    video_name = "VIRB0392.MP4"
    video = util.load_video(video_name)

    print('\n')
    print("Initializing histogram...")
    detect.init_histogram()
    print("Initializing ORB database...")
    retr.init_database()
    print("Reading data from CSV file...")
    retr.read_file()
    print("Done")

    global processing, finished
    processing = True
    finished = False

    detectThread = threading.Thread(target=detectThreadBody, args=(video, ))
    detectThread.start()

    rectifyThread = threading.Thread(target=rectifyThreadBody)
    rectifyThread.start()

    retrieveThread = threading.Thread(target=retrieveThreadBody)
    retrieveThread.start()

    peopleThread = threading.Thread(target=peopleThreadBody)
    peopleThread.start()

    global PeopleQueue, outputs
    while finished is False or len(outputs) > 0:
        n_frame = PeopleQueue.get()
        block = outputs[n_frame]
        del outputs[n_frame]

        print("\n-----------------------------------")
        print("\nFrame", block.n_frame)

        # Show results
        print("\nROI list:", block.roi_list)
        util.print_ranking(block.retrieved)
        util.print_room(block.room)
        util.draw(block.roi_list, block.cont_list, block.rectified, block.retrieved, block.people_boxes, block.room,
                  block.frame)

        # Delay & escape-key
        # video = skip_frames(video, fps=1)
        if cv2.waitKey(2) == ord('q'):
            processing = False
            break
        # cv2.waitKey()
    processing = False

    detectThread.join()
    rectifyThread.join()
    retrieveThread.join()
    peopleThread.join()
    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
