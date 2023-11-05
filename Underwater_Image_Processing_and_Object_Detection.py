import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
import platform
from typing import List, NamedTuple
import json
import cv2
import tensorflow as tf
import numpy as np
from tflite_support import metadata
from matplotlib import pyplot as plt
from streamlit_drawable_canvas import st_canvas
from matplotlib.patches import Rectangle

Interpreter = tf.lite.Interpreter
load_delegate = tf.lite.experimental.load_delegate


class ObjectDetectorOptions(NamedTuple):
    """A config to initialize an object detector."""

    enable_edgetpu: bool = False
    """Enable the model to run on EdgeTPU."""

    label_allow_list: List[str] = None
    """The optional allow list of labels."""

    label_deny_list: List[str] = None
    """The optional deny list of labels."""

    max_results: int = -1
    """The maximum number of top-scored detection results to return."""

    num_threads: int = 1
    """The number of CPU threads to be used."""

    score_threshold: float = 0.0
    """The score threshold of detection results to return."""


class Rect(NamedTuple):
    """A rectangle in 2D space."""
    left: float
    top: float
    right: float
    bottom: float


class Category(NamedTuple):
    """A result of a classification task."""
    label: str
    score: float
    index: int


class Detection(NamedTuple):
    """A detected object as the result of an ObjectDetector."""
    bounding_box: Rect
    categories: List[Category]


def edgetpu_lib_name():
    """Returns the library name of EdgeTPU in the current platform."""
    return {
        'Darwin': 'libedgetpu.1.dylib',
        'Linux': 'libedgetpu.so.1',
        'Windows': 'edgetpu.dll',
    }.get(platform.system(), None)


class ObjectDetector:
    """A wrapper class for a TFLite object detection model."""

    _OUTPUT_LOCATION_NAME = 'location'
    _OUTPUT_CATEGORY_NAME = 'category'
    _OUTPUT_SCORE_NAME = 'score'
    _OUTPUT_NUMBER_NAME = 'number of detections'

    def __init__(
        self,
        model_path: str,
        options: ObjectDetectorOptions = ObjectDetectorOptions()
    ) -> None:
        """Initialize a TFLite object detection model.
        Args:
            model_path: Path to the TFLite model.
            options: The config to initialize an object detector. (Optional)
        Raises:
            ValueError: If the TFLite model is invalid.
            OSError: If the current OS isn't supported by EdgeTPU.
        """

        # Load metadata from model.
        displayer = metadata.MetadataDisplayer.with_model_file(model_path)

        # Save model metadata for preprocessing later.
        model_metadata = json.loads(displayer.get_metadata_json())
        process_units = model_metadata['subgraph_metadata'][0]['input_tensor_metadata'
                                                               ][0]['process_units']
        mean = 0.0
        std = 1.0
        for option in process_units:
            if option['options_type'] == 'NormalizationOptions':
                mean = option['options']['mean'][0]
                std = option['options']['std'][0]
        self._mean = mean
        self._std = std

        # Load label list from metadata.
        file_name = displayer.get_packed_associated_file_list()[0]
        label_map_file = displayer.get_associated_file_buffer(
            file_name).decode()
        label_list = list(filter(lambda x: len(x) > 0,
                          label_map_file.splitlines()))
        self._label_list = label_list

        # Initialize TFLite model.
        if options.enable_edgetpu:
            if edgetpu_lib_name() is None:
                raise OSError(
                    "The current OS isn't supported by Coral EdgeTPU.")
            interpreter = Interpreter(
                model_path=model_path,
                experimental_delegates=[load_delegate(edgetpu_lib_name())],
                num_threads=options.num_threads)
        else:
            interpreter = Interpreter(
                model_path=model_path, num_threads=options.num_threads)

        interpreter.allocate_tensors()
        input_detail = interpreter.get_input_details()[0]

        sorted_output_indices = sorted(
            [output['index'] for output in interpreter.get_output_details()])
        self._output_indices = {
            self._OUTPUT_LOCATION_NAME: sorted_output_indices[0],
            self._OUTPUT_CATEGORY_NAME: sorted_output_indices[1],
            self._OUTPUT_SCORE_NAME: sorted_output_indices[2],
            self._OUTPUT_NUMBER_NAME: sorted_output_indices[3],
        }

        self._input_size = input_detail['shape'][2], input_detail['shape'][1]
        self._is_quantized_input = input_detail['dtype'] == np.uint8
        self._interpreter = interpreter
        self._options = options

    def detect(self, input_image: np.ndarray) -> List[Detection]:
        """Run detection on an input image.
        Args:
            input_image: A [height, width, 3] RGB image. Note that height and width
              can be anything since the image will be immediately resized according
              to the needs of the model within this function.
        Returns:
            A Person instance.
        """
        image_height, image_width, _ = input_image.shape

        input_tensor = self._preprocess(input_image)

        self._set_input_tensor(input_tensor)
        self._interpreter.invoke()

        # Get all output details
        boxes = self._get_output_tensor(self._OUTPUT_LOCATION_NAME)
        classes = self._get_output_tensor(self._OUTPUT_CATEGORY_NAME)
        scores = self._get_output_tensor(self._OUTPUT_SCORE_NAME)
        count = int(self._get_output_tensor(self._OUTPUT_NUMBER_NAME))

        return self._postprocess(boxes, classes, scores, count, image_width,
                                 image_height)

    def _preprocess(self, input_image: np.ndarray) -> np.ndarray:
        """Preprocess the input image as required by the TFLite model."""

        # Resize the input
        input_tensor = cv2.resize(input_image, self._input_size)

        # Normalize the input if it's a float model (aka. not quantized)
        if not self._is_quantized_input:
            input_tensor = (np.float32(input_tensor) - self._mean) / self._std

        # Add batch dimension
        input_tensor = np.expand_dims(input_tensor, axis=0)

        return input_tensor

    def _set_input_tensor(self, image):
        """Sets the input tensor."""
        tensor_index = self._interpreter.get_input_details()[0]['index']
        input_tensor = self._interpreter.tensor(tensor_index)()[0]
        input_tensor[:, :] = image

    def _get_output_tensor(self, name):
        """Returns the output tensor at the given index."""
        output_index = self._output_indices[name]
        tensor = np.squeeze(self._interpreter.get_tensor(output_index))
        return tensor

    def _postprocess(self, boxes: np.ndarray, classes: np.ndarray,
                     scores: np.ndarray, count: int, image_width: int,
                     image_height: int) -> List[Detection]:
        """Post-process the output of TFLite model into a list of Detection objects.
        Args:
            boxes: Bounding boxes of detected objects from the TFLite model.
            classes: Class index of the detected objects from the TFLite model.
            scores: Confidence scores of the detected objects from the TFLite model.
            count: Number of detected objects from the TFLite model.
            image_width: Width of the input image.
            image_height: Height of the input image.
        Returns:
            A list of Detection objects detected by the TFLite model.
        """
        results = []

        # Parse the model output into a list of Detection entities.
        for i in range(count):
            if scores[i] >= self._options.score_threshold:
                y_min, x_min, y_max, x_max = boxes[i]
                bounding_box = Rect(
                    top=int(y_min * image_height),
                    left=int(x_min * image_width),
                    bottom=int(y_max * image_height),
                    right=int(x_max * image_width))
                class_id = int(classes[i])
                category = Category(
                    score=scores[i],
                    # 0 is reserved for background
                    label=self._label_list[class_id],
                    index=class_id)
                result = Detection(bounding_box=bounding_box,
                                   categories=[category])
                results.append(result)

        # Sort detection results by score ascending
        sorted_results = sorted(
            results,
            key=lambda detection: detection.categories[0].score,
            reverse=True)

        # Filter out detections in deny list
        filtered_results = sorted_results
        if self._options.label_deny_list is not None:
            filtered_results = list(
                filter(
                    lambda detection: detection.categories[0].label not in self.
                    _options.label_deny_list, filtered_results))

        # Keep only detections in allow list
        if self._options.label_allow_list is not None:
            filtered_results = list(
                filter(
                    lambda detection: detection.categories[0].label in self._options.
                    label_allow_list, filtered_results))

        # Only return maximum of max_results detection.
        if self._options.max_results > 0:
            result_count = min(len(filtered_results),
                               self._options.max_results)
            filtered_results = filtered_results[:result_count]

        return filtered_results


_MARGIN = 10  # pixels
_ROW_SIZE = 10  # pixels
_FONT_SIZE = 1  # text size
_FONT_THICKNESS = 1  # text thickness
_TEXT_COLOR = (0, 0, 0)  # color
_BOX_COLOR = (64, 235, 52)  # color


def visualize(
    image: np.ndarray,
    detections: List[Detection],
    text_color, box_color
) -> np.ndarray:
    """Draws bounding boxes on the input image and return it.
    Args:
      image: The input RGB image.
      detections: The list of all "Detection" entities to be visualize.
    Returns:
      Image with bounding boxes.
    """
    for detection in detections:
        # Draw bounding_box
        start_point = detection.bounding_box.left, detection.bounding_box.top
        end_point = detection.bounding_box.right, detection.bounding_box.bottom
        cv2.rectangle(image, start_point, end_point, box_color, 2)

        # Draw label and score
        category = detection.categories[0]
        class_name = category.label
        probability = round(category.score, 2)
        result_text = class_name + ' (' + str(probability) + ')'
        text_location = (_MARGIN + detection.bounding_box.left,
                         _MARGIN + _ROW_SIZE + detection.bounding_box.top)
        cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    _FONT_SIZE, text_color, _FONT_THICKNESS)

    return image


def objectDetector(image, DETECTION_THRESHOLD=0.5,
                    text_color=(0, 0, 0),
                    box_color=(34, 143, 34)):
    
    TFLITE_MODEL_PATH = "./model/model2.tflite"

    # image = Image.open(TEMP_FILE).convert('RGB')
    image.thumbnail((512, 512), Image.ANTIALIAS)
    image_np = np.asarray(image)

    # Load the TFLite model
    options = ObjectDetectorOptions(
        num_threads=8,
        score_threshold=DETECTION_THRESHOLD,
    )
    detector = ObjectDetector(model_path=TFLITE_MODEL_PATH, options=options)

    # Run object detection estimation using the model.
    detections = detector.detect(image_np)

    # count of objects detected
    count = len(detections)
    score = 0
    for detected in detections:
        category = detected.categories[0]
        score += round(category.score, 2)

    # Draw keypoints and edges on input image
    image_np = visualize(image_np, detections, text_color, box_color)

    # Show the detection result
    if count!=0:
        return Image.fromarray(image_np), count, score/count
    else:
        return Image.fromarray(image_np), count, 0


def whitepatch_balancing(image, from_row, from_column, row_width, column_width):
    image_patch = image[from_row:from_row+row_width,
                        from_column:from_column+column_width]
    image_max = (image*1.0 / image_patch.max(axis=(0, 1))).clip(0, 1)
    white_balanc_image = Image.fromarray((image_max * 255).astype(np.uint8))
    image1 = Image.fromarray((image_max * 255).astype(np.uint8), mode="RGB")
    img = ImageEnhance.Sharpness(image1)
    img.enhance(2).save("./tmp/image.png")
    return white_balanc_image, Image.open("./tmp/image.png")

"""## UNDERWATER IMAGE CLASSIFICATION USING DEEP LEARNING"""
uploaded_file = st.file_uploader("Choose a **Image** file", type=".jpg")
if uploaded_file:
    "### Image Processing..."
    selected_image = Image.open(uploaded_file)
    selected_image = selected_image.resize([512, 512])
    st.columns((2, 13, 2))[1].image(selected_image, caption='Original Image')
    _, col, _ = st.columns((2, 13, 2))
    with col:
        canvas = st_canvas(
            fill_color="#ffffff00",
            stroke_width=2,
            stroke_color="#E53935",
            background_image=selected_image,
            drawing_mode="rect",
            point_display_radius=0,
            height=512,
            width=512,
        )
    # st.write(canvas.json_data['objects'])

    open_cv_image = np.array(selected_image)
    if canvas.json_data['objects']:
        # white_balancing
        white_balanc_image, enhanced_image = whitepatch_balancing(
            open_cv_image,
            canvas.json_data['objects'][len(
                canvas.json_data['objects'])-1]['left'],
            canvas.json_data['objects'][len(
                canvas.json_data['objects'])-1]['top'],
            canvas.json_data['objects'][len(
                canvas.json_data['objects'])-1]['width'],
            canvas.json_data['objects'][len(canvas.json_data['objects'])-1]['height'])

        col1, col2 = st.columns(2)
        with col1:
            st.image(white_balanc_image, caption='white balanc Image')
        with col2:
            enhanced_image = enhanced_image.filter(ImageFilter.SMOOTH)
            st.image(enhanced_image, caption='Enhanced Image')

        values = st.slider(
            'Select threshold value',
            0.0, 1.0, 0.5)

        col1, col2, col3 = st.columns(3)
        with col1:
            text_color = st.color_picker("Select text color", value='#000000')
        with col2:
            box_color = st.color_picker("Select box color", value='#228f22')

        h = text_color.lstrip('#')
        text_color = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
        h1 = box_color.lstrip('#')
        box_color = tuple(int(h1[i:i+2], 16) for i in (0, 2, 4))

        enhanced_image, enhanced_object_count, enhanced_object_score = objectDetector(
            enhanced_image, values, text_color, box_color)

        original_image, original_object_count, original_object_score = objectDetector(
            selected_image, values, text_color, box_color)

        col1, col2 = st.columns(2)
        with col1:
            st.image(original_image, caption='Original Image')
            st.write("+ Objects count: **"+str(original_object_count)+"**")
            try:
                st.write("+ Objects score average :**" +
                         str(original_object_score*100)[:5]+'**%')
            except:
                st.write('+ Objects score average :**' +
                         str(original_object_score*100)+'**%')

        with col2:
            st.image(enhanced_image, caption='Enhanced Image')
            st.write("+ Objects count: **"+str(enhanced_object_count)+"**")
            try:
                st.write("+ Objects score average :**" +
                         str(enhanced_object_score*100)[:5]+"**%")
            except:
                st.write("+ Objects score average :**" +
                         str(enhanced_object_score*100)+"**%")

        # st.columns((2, 13, 2))[1].image(original_image, caption='Original Image')
        # st.columns((2, 13, 2))[1].image(enhanced_image, caption='Enhanced Image')


else:
    "please select **Image** file..."