import os
# import a utility function for loading Roboflow models
from inference import get_roboflow_model
# import supervision to visualize our results
import supervision as sv
# import cv2 to helo load our image
import cv2

api_key="Ruhj6rAp7XcoDVv7c5u1"
os.environ["ROBOFLOW_API_KEY"] = api_key

# define the image url to use for inference
image_file = f"{os.environ['HOME']}/Videos/sharksbb1-1/panorama.tif"
image = cv2.imread(image_file)

# load a pre-trained yolov8n model
model = get_roboflow_model(model_id="ice-rink-osa5z/2")

# run inference on our chosen image, image can be a url, a numpy array, a PIL image, etc.
results = model.infer(image)

# load the results into the supervision Detections api
detections = sv.Detections.from_inference(results[0].dict(by_alias=True, exclude_none=True))

# create supervision annotators
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

# annotate the image with our inference results
annotated_image = bounding_box_annotator.annotate(
    scene=image, detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections)

# display the image
sv.plot_image(annotated_image)
