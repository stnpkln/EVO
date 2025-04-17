window_3x3 = {
	"name": "window_3x3",
	"coords": [
		[-1, -1], # levy horni roh
		[-1, 0], # horni stred
		[-1, 1], # pravy horni roh
		[0, -1], # levy stred
		[0, 0], # stred
		[0, 1], # pravy stred
		[1, -1], # levy dolni roh
		[1, 0], # dolni stred
		[1, 1] # pravy dolni roh
	]
}

def apply_window(image, window, x, y):
	"""
	Applies a window to a pixel in an image.
	:param image: The input image as a 2D array.
	:param window: The window to apply.
	:param x: The x-coordinate of the pixel.
	:param y: The y-coordinate of the pixel.
	:return: The result of applying the window to the pixel.
	"""
	window_coords = window["coords"] # we dont need the name here
	window_size = len(window_coords)
	result = [0] * window_size # Initialize result with zeros

	for (x_off, y_off) in window_coords:
		# Calculate the new coordinates
		new_x = x + x_off
		new_y = y + y_off

		# Check if the new coordinates are within the image bounds
		if 0 <= new_x < len(image) and 0 <= new_y < len(image[0]):
			result.append(image[new_x][new_y])

	return result