import numpy as np
import cv2
import painting_detection as detect
import painting_retrieval as retr
import argparse
import time

fps = 1
n_frame = 0
time_now = 0


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-vid', metavar='video_name',
                        help="name of the video to be analyzed. If not present, the 'VIRB0392.MP4' video will be \
                             analyzed (if present in the '<code-directory>/videos' folder)")
    parser.add_argument('-frame', metavar='frame_number', type=int,
                        help="frame number indicating where to start. Default 0")
    parser.add_argument('-fps', metavar='fps_number', type=int,
                        help="number of frame to analyze every second, skipping those in between. " +
                             "If not present, all frames will be analyzed")
    args = parser.parse_args()

    video_name = args.vid
    num_frame = args.frame
    n_fps = args.fps

    if num_frame is not None and num_frame < 0:
        print("frame_number must be an integer grater or equal to zero")
        exit(1)

    if n_fps is not None and n_fps < 1:
        print("fps_number must be an integer grater or equeal to one")
        exit(1)

    return video_name, n_frame, n_fps


def load_video(video_name, num_frame):
    path = "videos/"
    video = cv2.VideoCapture(path + video_name)
    if not video.isOpened():
        print("File", path + video_name, "not found.")
        exit(1)
    video.set(cv2.CAP_PROP_POS_FRAMES, num_frame)
    return video


def init(default_video, start_frame=0, default_fps=None):
    global fps, n_frame

    video_name, n_frame, fps = get_args()
    if video_name is None:
        video_name = default_video
    if n_frame is None:
        n_frame = start_frame
    if fps is None:
        fps = default_fps
    print("\nLoading video...")
    video = load_video(video_name, n_frame)
    print("Initializing histogram...")
    detect.init_histogram()
    print("Initializing ORB database...")
    retr.init_database()
    print("Reading data from CSV file...")
    retr.read_file()
    print("Done")

    return video


def video_end(video):
    return not video.grab()


def get_next_frame(video):
    global time_now, fps, n_frame

    time_now = time.time()
    _, frame = video.retrieve()
    n_frame = int(video.get(cv2.CAP_PROP_POS_FRAMES)) - 1
    if fps is not None:
        go_to = int(video.get(cv2.CAP_PROP_POS_FRAMES)) + int(video.get(cv2.CAP_PROP_FPS)) / fps
        last = video.get(cv2.CAP_PROP_FRAME_COUNT) - 1
        if go_to <= last:
            video.set(cv2.CAP_PROP_POS_FRAMES, go_to)
        else:
            video.set(cv2.CAP_PROP_POS_FRAMES, last)
            video.grab()
    return frame


def resize_images(paintings):
    small_paintings = []
    for painting in paintings:
        h, w = painting.shape[0:2]
        if h > w:
            wide = h
            border = int((h - w) / 2)
            vert = True
        else:
            wide = w
            border = int((w - h) / 2)
            vert = False
        small = np.zeros((wide, wide, 3), dtype=np.uint8)
        if vert:
            if border * 2 != np.abs(h - w):
                small[:, border:wide - border - 1] = painting
            else:
                small[:, border:wide - border] = painting
        else:
            if border * 2 != np.abs(h - w):
                small[border:wide - border - 1] = painting
            else:
                small[border:wide - border] = painting
        size = int(1280 / 4)
        small = cv2.resize(small, (size, size))
        small_paintings.append(small)
    return small_paintings


def concatenate_rectified_retrieval(rect_concat, retrieved):
    retr_img0 = get_retrieved_img(retrieved, ranking=0)
    retr_img1 = get_retrieved_img(retrieved, ranking=1)
    retr_img2 = get_retrieved_img(retrieved, ranking=2)

    retr_img0 = resize_images(retr_img0)
    retr_img1 = resize_images(retr_img1)
    retr_img2 = resize_images(retr_img2)

    retr_concat0 = np.concatenate(retr_img0, axis=1)
    retr_concat1 = np.concatenate(retr_img1, axis=1)
    retr_concat2 = np.concatenate(retr_img2, axis=1)

    concatenate = np.concatenate((rect_concat, retr_concat0, retr_concat1, retr_concat2), axis=0)
    return concatenate


def get_retrieved_img(retrieved, ranking=0):
    retr_img = []
    retr_names = retrieved[:, ranking, 0]
    for name in retr_names:
        img = retr.paintings_db[name]
        retr_img.append(img)
    return retr_img


def print_ranking(retrieved):
    print("\nDatabase matches:")
    for i, img_retr in enumerate(retrieved):
        print(f"Painting {i + 1}")
        for j, match in zip(range(3), img_retr):
            img_name, accuracy = match
            accuracy = round(np.float32(accuracy))
            title, author, _ = retr.data_csv[img_name]
            print(f"{j + 1} - {title}, {author}: {accuracy}%")


def print_room(room):
    if room == 0:
        print("\nNo room found.")
    else:
        print("\nRoom:", room)


def draw(roi_list, cont_list, rectified, retrieved, people_boxes, room, frame):
    roi_frame = np.array(frame)
    for rect in roi_list:
        (x, y, w, h) = rect
        cv2.rectangle(roi_frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
    for box in people_boxes:
        (x, y, w, h) = box
        cv2.rectangle(roi_frame, (x, y), (x + w, y + h), (255, 50, 0), thickness=2)
    if room == 0:
        room = "No room found"
    else:
        room = 'Room ' + str(room)
    cv2.putText(roi_frame, room, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 50, 255), thickness=2)

    mask = np.zeros_like(frame)
    for cont in cont_list:
        cv2.drawContours(mask, [cont], -1, (255, 255, 255), thickness=cv2.FILLED)
    segm_frame = np.uint8(mask == 255) * frame

    vertical_concat = np.concatenate((roi_frame, segm_frame), axis=0)

    if vertical_concat.shape[0] > 2000:
        fx, fy = 0.45, 0.45
    else:
        fx, fy = 0.65, 0.65
    cv2.namedWindow("Detection & Segmentation",
                    flags=cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
    cv2.imshow("Detection & Segmentation", cv2.resize(vertical_concat, None, fx=fx, fy=fy))

    if len(rectified) == 0:
        return
    small_rectified = resize_images(rectified)
    rect_concat = np.concatenate(small_rectified, axis=1)

    if len(retrieved) > 0:
        concatenate = concatenate_rectified_retrieval(rect_concat, retrieved)
    else:
        concatenate = rect_concat

    cv2.namedWindow("Painting Rectification & Retrieval",
                    flags=cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
    cv2.imshow("Painting Rectification & Retrieval", cv2.resize(concatenate, None, fx=0.75, fy=0.75))


def show_results(frame, roi_list, cont_list, rectified, retrieved, room, people_boxes):
    global time_now, n_frame
    print("\n-----------------------------------")
    print("\nFrame", n_frame)
    print("\nROI list:", roi_list)
    print_ranking(retrieved)
    print_room(room)
    draw(roi_list, cont_list, rectified, retrieved, people_boxes, room, frame)
    print("\nFrame computation time =", time.time() - time_now)


def close_all(video):
    video.release()
    cv2.destroyAllWindows()
