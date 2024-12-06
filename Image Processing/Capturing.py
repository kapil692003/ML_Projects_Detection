import cv2

# Frame Adjustment according to madel dimension
separator_width = 450  # in pixels
separator_height = 250  # in pixels


cap = cv2.VideoCapture(0)

# Get the width and height of the camera feed
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Calculate the top-left corner of the separator frame
frame_x = (frame_width - separator_width) // 2
frame_y = (frame_height - separator_height) // 2

# Window Size of camera Capture
capture_window_width = 1200
capture_window_height = 900


cv2.namedWindow('Align Separator and Press "s" to Capture', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Align Separator and Press "s" to Capture', capture_window_width, capture_window_height)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Draw the separator frame on the live feed
    cv2.rectangle(frame, (frame_x, frame_y), (frame_x + separator_width, frame_y + separator_height), (0, 255, 0), 2)

    # Display instructions on the live feed
    cv2.putText(frame, 'Align separator within the frame', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, 'Press "s" to Capture, "q" to Quit', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Display the resulting frame
    cv2.imshow('Align Separator and Press "s" to Capture', frame)

    # Wait for key press
    key = cv2.waitKey(1)

    # If 's' is pressed, capture the image
    if key == ord('s'):
        # Crop the frame to the size of the separator
        captured_image = frame[frame_y:frame_y + separator_height, frame_x:frame_x + separator_width]

        # Save the captured image
        cv2.imwrite('captured_separator.jpg', captured_image)
        print("Image captured and saved as 'captured_separator.jpg'")
        
        # Proceed to process the captured image
        break

    # If 'q' is pressed, quit without capturing
    elif key == ord('q'):
        print("Quit without capturing.")
        break

# Release the camera
cap.release()
cv2.destroyAllWindows()

# Load the captured image
image = cv2.imread('captured_separator.jpg')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# edge detection
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
    cv2.putText(image, f"Height from bottom: {height_from_bottom}px", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


cv2.namedWindow('Processed Image with Liquid Height', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Processed Image with Liquid Height', capture_window_width, capture_window_height)


cv2.imshow('Processed Image with Liquid Height', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
