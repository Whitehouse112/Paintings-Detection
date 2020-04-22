
video = cv2.VideoCapture('VIRB0407.MP4')    # folder 001
if not video.isOpened():
    print("File not found.")

frame_entr = 6.5
while video.grab():
    _, frame = video.retrieve()

    blurred = cv2.cvtColor(cv2.GaussianBlur(frame, (3, 3), 0), cv2.COLOR_RGB2GRAY)
    grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_ISOLATED)
    grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_ISOLATED)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    cv2.imshow("sobel", grad)

    grad = (grad < 15).astype(np.uint8) * 255
    grad = cv2.morphologyEx(cv2.erode(grad, np.ones((1, 1), np.uint8), iterations=2), cv2.MORPH_CLOSE, (4, 4), iterations=1)
    cv2.imshow("sobel2", grad)


    contours, _ = cv2.findContours(grad, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(contours)
    img_contours = np.zeros_like(frame)
    cv2.drawContours(img_contours, contours, -1, (0, 255, 0), thickness=2)
    cv2.drawContours(frame, contours, -1, (0, 255, 0), thickness=2)

    cv2.imshow("frame", frame)


    if cv2.waitKey(1) == ord('q'):
            break

video.release()
cv2.destroyAllWindows()