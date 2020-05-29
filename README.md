# Vc-Group
Final project for (Computer) Vision and Cognitive System

## Requirements
- numpy
- opencv-python

## Pipeline
For each video frame:

### Painting detection & segmentation
Predict a ROI for each painting:
1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
2. Edge Detection with Sobel
3. Bilateral Filtering
4. Thresholding
5. Morphology Transformations
6. Significant Contours (cv2.findContours)
7. Contours refining:
   - Find Bounding Boxes (cv2.boundingRect)
   - Merge overlapping
   - Convex hull
8. Discard false positives:
   - Check dimensions and aspect ration
   - Histogram distance & update

### Painting rectification
Starting from contours found in previous point and considering one contour at a time:
1. Polygonal approximation (cv2.approxPolyDP)
2. Find lines with Hough transform
3. Compute lines intersections
4. Average vertices with K-Means
5. Order vertices
6. Compute aspect-ratio
7. Warp perspective

### Painting retrieval & localization
Match each detected painting to the paintings DB:
1. Find descriptors with ORB
2. Find best matches (BFMatcher with Hamming normalization)
3. Find room in which paintings are collocated

### People detection
Predict a ROI around each person:
1. YOLO v3 (from OpenCV)
   
   For each detection:
   - Predict a score for each class
   - Take only the class corresponding to the best score
   - Take only matches belonging to the person class
   - Thresholding
   - Non-maximum suppression
2. Discard people in paintings
