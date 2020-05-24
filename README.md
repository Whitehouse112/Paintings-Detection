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
7. Contours checking:
   - Find Bounding Boxes (cv2.boundingRect)
   - Merge overlapping
   - Convex hull
8. Discard false positives:
   - Check dimensions and aspect ration
   - Histogram distance & update

### Painting rectification
Starting from contours found in previous point and considering one contour at a time:
1. ApproxPolyDP
2. Hough Lines
3. Find lines intersections
4. K-means to find average vertices
5. Order vertices
6. Compute aspect-ratio
7. Warp perspective

### Painting retrieval
Match each detected painting to the paintings DB:
1. ORB
