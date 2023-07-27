import numpy as np
import tensorflow as tf
import cv2
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


class Inference:
    def __init__(self):
        self.model = tf.saved_model.load("saved_model")
        self.category_index = label_map_util.create_category_index_from_labelmap("utils/label_map.pbtxt",
                                                                                 use_display_name=True)
        self.output_image_path = 'tempdir/prediction.png'

    def run_inference_for_single_image(self, image):
        # Image preprocessing
        image = np.asarray(image)
        input_tensor = tf.convert_to_tensor(image)
        input_tensor = input_tensor[tf.newaxis, ...]
        # Inference
        model_fn = self.model.signatures['serving_default']
        output_dict = model_fn(input_tensor)
        # Convert to numpy arrays
        num_detections = int(output_dict.pop('num_detections'))
        output_dict = {key: value[0, :num_detections].numpy() for key, value in output_dict.items()}
        output_dict['num_detections'] = num_detections
        # detection_classes should be ints.
        output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
        return output_dict

    def save_image_with_bboxes(self, image, output_dict):
        image_np_with_detections = image.copy()
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            self.category_index,
            instance_masks=output_dict.get('detection_masks_reframed', None),
            use_normalized_coordinates=True,
            line_thickness=8)
        cv2.imwrite(self.output_image_path, cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))

    def __call__(self, image_path):
        # Read the image from the file
        image = cv2.imread(image_path)
        # Perform inference on the image
        output_dict = self.run_inference_for_single_image(image)
        # Save the image with bounding boxes
        self.save_image_with_bboxes(image, output_dict)
        for key in output_dict.keys():
            try:
                output_dict[key] = output_dict[key].tolist()
            except Exception as e:
                pass
        return output_dict

