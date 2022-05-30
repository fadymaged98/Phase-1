our target is to get lines that contours the car road 
-	To deal with image, we use open cv to make equations and draw (show ) results
-	First step ( image effects) 
1-	Grayscale 
2-	Gaussian Blur
3-	Get lines using low and high threshold

-	Second step (region of interest):
Show only pixels we want to deal with by:
Creating mask (triangle)
Applying this mask on the output from step one

-	Third step (get lines)
According to Hough transform, we can transform the points from image to a line passing through it

Last thing is to apply this lines on our image.


