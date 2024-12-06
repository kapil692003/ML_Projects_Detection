import cv2
import numpy as np

# Load the image
image = cv2.imread("photo4.jpg") 
image = cv2.resize(image,(700,500))
# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Perform edge detection
edges = cv2.Canny(blurred, 50, 150)

# Find contours in the edge map
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sort contours by their vertical position (top to bottom)
contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])

# Get the height of the image
image_height = image.shape[0]

# Draw lines and calculate height from bottom
for contour in contours:
    # Get the bounding box of the contour
    x, y, w, h = cv2.boundingRect(contour)
    
    # Calculate the top of the liquid level (y-coordinate of the bounding box)
    liquid_top_y = y
    
    # Calculate the height of the liquid from the bottom
    height_from_bottom = image_height - liquid_top_y
    
    # Define the start and end points of the line segment (adjust the length as needed)
    start_point = (x, y + h // 2)
    end_point = (x + w, y + h // 2)
    
    # Draw the line segment
    cv2.line(image, start_point, end_point, (0, 255, 0), 2)
    
    # Display the height from the bottom on the image
    cv2.putText(image, f"H: {height_from_bottom}px", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# Display the result
cv2.imshow('Liquid Levels with Height', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
