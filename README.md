# Vc-Group
Final project for (Computer) Vision and Cognitive System

## Pipeline

For each frame of the video:

### Painting detection
1. Gaussian Blurring
2. Edge Detection with Sobel
3. Morphology Transformations
4. Significant Contours (cv2.findContours)
5. Find Bounding Boxes (cv2.boundingRect)
6. Discard false positives:
   - Check dimensions and aspect ration
   - Histogram distance
   - Discard inner rectangles

### Painting rectification
Starting from contours found in previous point and considering one contour at a time:
1. Convex hull
2. ApproxPolyDP
3. Hough Lines
4. Find lines intersections
5. K-means
6. Warp perspective
