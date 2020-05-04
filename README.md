# Vc-Group
Final project for (Computer) Vision and Cognitive System

## Pipeline

Painting detection:

For each frame of the video:
1. Gaussian Blurring
2. Edge Detection with Sobel
3. Morphology Transformations
4. Significant Contours
5. Find Bounding Boxes (cv2.boundingRect)
6. Discard false positives
   - Check dimensions and aspect ration
   - Histogram distance
   - Discard inner rectangles
