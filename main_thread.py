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
import threading
import queue
import time


# ------------------------------------------------------------------------
# GLOBAL VARIABLES
# ------------------------------------------------------------------------
processing = None
finished = None
outputs = {}
DetectQueue = queue.Queue()
RectifyQueue = queue.Queue()
RetrieveQueue = queue.Queue()
PeopleQueue = queue.Queue()


# ------------------------------------------------------------------------
# GLOBAL DATA STRUCTURES
# ------------------------------------------------------------------------
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


# ------------------------------------------------------------------------
# PAINTING DETECTION THREAD
# ------------------------------------------------------------------------
def detectThreadBody(video):
    global DetectQueue, processing, outputs, finished

    while video.grab() and processing:
        _, frame = video.retrieve()  # (H, W, 3)

        while processing and DetectQueue.qsize() > 30:
            pass

        block = Frame(frame)
        block.n_frame = int(video.get(cv2.CAP_PROP_POS_FRAMES)) - 1

        time_now_detection = time.time()
        block.roi_list, block.cont_list = detect.detect_paintings(block.frame)
        block.time_detection = time.time() - time_now_detection

        outputs[block.n_frame] = block
        DetectQueue.put(block.n_frame)

        # video = util.skip_frames(video, fps=2)
    finished = True


# ------------------------------------------------------------------------
# PAINTING RECTIFICATION THREAD
# ------------------------------------------------------------------------
def rectifyThreadBody():
    global DetectQueue, RectifyQueue, outputs, processing

    while processing:
        while processing and RectifyQueue.qsize() > 30:
            pass

        try:
            n_frame = DetectQueue.get(timeout=1)
            block = outputs[n_frame]

            time_now_rectification = time.time()
            block.rectified, block.roi_list, block.cont_list = rect.rectify_paintings(block.roi_list, block.cont_list,
                                                                                      block.frame)
            block.time_rectification = time.time() - time_now_rectification

            outputs[block.n_frame] = block
            RectifyQueue.put(block.n_frame)
        except queue.Empty:
            pass


# ------------------------------------------------------------------------
# PAINTING RETRIEVAL THREAD
# ------------------------------------------------------------------------
def retrieveThreadBody():
    global RectifyQueue, RetrieveQueue, outputs, processing

    while processing:
        while processing and RetrieveQueue.qsize() > 30:
            pass

        try:
            n_frame = RectifyQueue.get(timeout=1)
            block = outputs[n_frame]

            time_now_retrieve = time.time()
            block.room, block.retrieved = retr.retrieve_paintings(block.rectified)
            block.time_retrieval = time.time() - time_now_retrieve

            outputs[block.n_frame] = block
            RetrieveQueue.put(block.n_frame)
        except queue.Empty:
            pass


# ------------------------------------------------------------------------
# PEOPLE DETECTION THREAD
# ------------------------------------------------------------------------
def peopleThreadBody():
    global RetrieveQueue, PeopleQueue, outputs, processing

    while processing:
        while processing and PeopleQueue.qsize() > 30:
            pass

        try:
            n_frame = RetrieveQueue.get(timeout=1)
            block = outputs[n_frame]

            time_now_people = time.time()
            block.people_boxes = people.detect_people(block.frame, block.roi_list)
            block.time_people = time.time() - time_now_people

            outputs[block.n_frame] = block
            PeopleQueue.put(block.n_frame)
        except queue.Empty:
            pass


# ------------------------------------------------------------------------
# MAIN process
# ------------------------------------------------------------------------
def main():
    global processing, finished, DetectQueue, RectifyQueue, RetrieveQueue, PeopleQueue, outputs

    video_name = "GOPR5826.MP4"
    video = util.init(video_name)

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

    time_frame = time.time()
    while finished is False or len(outputs) > 0:
        try:
            n_frame = PeopleQueue.get(timeout=5)
            block = outputs[n_frame]
            del outputs[n_frame]

            time_now_draw = time.time()
            util.show_results(block)

            # Timing performances
            block.time_draw = time.time() - time_now_draw
            time_elapsed = time.time() - time_frame
            total_time = block.time_detection + block.time_rectification + block.time_retrieval + \
                block.time_people + block.time_draw
            print("\nDetection time = ", block.time_detection, DetectQueue.qsize())
            print("Rectification time = ", block.time_rectification, RectifyQueue.qsize())
            print("Retrieve time = ", block.time_retrieval, RetrieveQueue.qsize())
            print("People time = ", block.time_people, PeopleQueue.qsize())
            print("Draw time = ", block.time_draw)
            print("\nTotal computation time = ", total_time)
            print("Threaded time = ", time_elapsed)
            time_frame = time.time()
        except queue.Empty:
            pass

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


# ------------------------------------------------------------------------
# STARTING POINT
# ------------------------------------------------------------------------
if __name__ == "__main__":
    main()

# ------------------------------------------------------------------------
