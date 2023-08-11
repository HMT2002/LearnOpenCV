
# #This is a comment.
# fruits = ["apple", "banana", "cherry"]
# x, y, z = fruits
# print(x)
# print(y)
# print(z)

# def myfunc():
#   x = "fantastic"
#   print("Python is " + x)

# myfunc()


import os
import mediapipe as mp
import cv2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red


def visualize(
    image,
    detection_result
) -> np.ndarray:
  """Draws bounding boxes on the input image and return it.
  Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualize.
  Returns:
    Image with bounding boxes.
  """
  for detection in detection_result.detections:
    # Draw bounding_box
    bbox = detection.bounding_box
    start_point = bbox.origin_x, bbox.origin_y
    end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
    cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

    # Draw label and score
    category = detection.categories[0]
    category_name = category.category_name
    probability = round(category.score, 2)
    result_text = category_name + ' (' + str(probability) + ')'
    text_location = (MARGIN + bbox.origin_x,
                     MARGIN + ROW_SIZE + bbox.origin_y)
    cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

  return image



file_path=os.path.abspath(__file__)
print('file_path is:'+file_path)

dir_path=os.path.abspath(__file__)
print('dir_path is:'+os.path.dirname(dir_path))
print('models_path is:'+os.path.dirname(dir_path)+'\efficientdet_lite0.tflite')

model_path = 'efficientdet_lite0.tflite'

# Load the input image from an image file.
imgaes_path=os.path.dirname(os.path.dirname(dir_path)) +'\\ffmpeg\\test.png'
print('images_path is:'+imgaes_path);
mp_image = mp.Image.create_from_file(imgaes_path)
print(mp_image)


BaseOptions = mp.tasks.BaseOptions
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    max_results=5,
    running_mode=VisionRunningMode.IMAGE)

detector = vision.ObjectDetector.create_from_options(options)

detection_result = detector.detect(mp_image)
print(detection_result);
f = open("detection_result.txt", "w")
f.write(str(detection_result))
f.close()

image_copy = np.copy(mp_image.numpy_view())
annotated_image = visualize(image_copy, detection_result)
rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
cv2.imshow('Image Window',rgb_annotated_image)

cv2.waitKey(0)





