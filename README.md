# Vc-Group
Final project for (Computer) Vision and Cognitive System

## Requirements
- numpy
- opencv-python

## Pipeline

For each video frame:

### Painting detection
Predict a ROI for each painting:
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
5. K-means to find average vertices
6. Order vertices
7. Compute aspect-ratio
8. Warp perspective

### Painting retrieval
Match each detected painting to the paintings DB:
1. ORB
