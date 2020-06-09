import cv2
import painting_detection as detect
import painting_rectification as rect
import painting_retrieval as retr
import people_detection as people
import utility as util
import threading
import queue
import time


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
    time_detection = 0
    time_rectification = 0
    time_retrieval = 0
    time_people = 0
    time_draw = 0

    def __init__(self, frame):
        self.frame = frame


def detectThreadBody(video):
    global DetectQueue, processing, outputs, finished

    while video.grab() and processing:
        _, frame = video.retrieve()  # (H, W, 3)

        # while processing and DetectQueue.qsize() > 30:
        #     pass

        block = Frame(frame)
        block.n_frame = int(video.get(cv2.CAP_PROP_POS_FRAMES)) - 1

        time_now = time.time()
        block.roi_list, block.cont_list = detect.detect_paintings(block.frame)
        block.time_detection = time.time() - time_now

        outputs[block.n_frame] = block
        DetectQueue.put(block.n_frame)

        # video = util.skip_frames(video, fps=2)
    finished = True


def rectifyThreadBody():
    global DetectQueue, RectifyQueue, outputs, processing

    while processing:
        # while processing and RectifyQueue.qsize() > 1000:
        #     pass

        try:
            n_frame = DetectQueue.get(timeout=1)
            block = outputs[n_frame]

            time_now = time.time()
            block.rectified, block.roi_list, block.cont_list = rect.rectify_paintings(block.roi_list, block.cont_list,
                                                                                      block.frame)
            block.time_rectification = time.time() - time_now

            outputs[block.n_frame] = block
            RectifyQueue.put(block.n_frame)
        except queue.Empty:
            continue


def retrieveThreadBody():
    global RectifyQueue, RetrieveQueue, outputs, processing

    while processing:
        # while processing and RetrieveQueue.qsize() > 1000:
        #     pass

        try:
            n_frame = RectifyQueue.get(timeout=1)
            block = outputs[n_frame]

            time_now = time.time()
            block.room, block.retrieved = retr.retrieve_paintings(block.rectified)
            block.time_retrieval = time.time() - time_now

            outputs[block.n_frame] = block
            RetrieveQueue.put(block.n_frame)
        except queue.Empty:
            continue


def peopleThreadBody():
    global RetrieveQueue, PeopleQueue, outputs, processing

    while processing:
        # while processing and PeopleQueue.qsize() > 1000:
        #     pass

        try:
            n_frame = RetrieveQueue.get(timeout=1)
            block = outputs[n_frame]

            time_now = time.time()
            block.people_boxes = people.detect_people(block.frame, block.roi_list)
            block.time_people = time.time() - time_now

            outputs[block.n_frame] = block
            PeopleQueue.put(block.n_frame)
        except queue.Empty:
            continue


def main():
    video_name = "GOPR5826.MP4"
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
    time_frame = time.time()
    while finished is False or len(outputs) > 0:
        n_frame = PeopleQueue.get()
        block = outputs[n_frame]
        del outputs[n_frame]

        time_now = time.time()
        print("\n-----------------------------------")
        print("\nFrame", block.n_frame)

        # Show results
        print("\nROI list:", block.roi_list)
        util.print_ranking(block.retrieved)
        util.print_room(block.room)
        util.draw(block.roi_list, block.cont_list, block.rectified, block.retrieved, block.people_boxes, 
                    block.room,block.frame)
        block.time_draw = time.time() - time_now
        time_elapsed = time.time() - time_frame
        print("Total computation time = ", block.time_detection + block.time_rectification + block.time_retrieval + block.time_people + block.time_draw)
        print("Detection time = ", block.time_detection)
        print("Rectification time = ", block.time_rectification)
        print("Retrieve time = ", block.time_retrieval)
        print("People time = ", block.time_people)
        print("Draw time = ", block.time_draw)
        print("\nThreaded time = ", time_elapsed)
        time_frame = time.time()

        # Delay & escape-key
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
