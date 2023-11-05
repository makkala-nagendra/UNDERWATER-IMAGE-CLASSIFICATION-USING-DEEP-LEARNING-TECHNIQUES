import os
import xml.etree.ElementTree as ET
import glob
import tensorflow as tf
import numpy as np
import cv2

img_height=360
img_width=480

def convert_annotation(yolo_annotation_path, image_folder_path, 
                       output_folder_path, class_names):
    # Open the YOLO annotation file
    with open(yolo_annotation_path, 'r') as f:
        lines = f.readlines()

    image_path=image_folder_path
    img = cv2.imread(image_path)
    img = cv2.resize(img, (img_height, img_width))
    image_width, image_height, _ = img.shape

    # Create an XML file with the same name as the YOLO annotation file
    xml_file_path = os.path.join(output_folder_path, 
                                 os.path.splitext(os.path.basename(yolo_annotation_path))[0] + '.xml')
    root = ET.Element('annotation')
    ET.SubElement(root, 'folder').text = output_folder_path
    ET.SubElement(root, 'filename').text = os.path.basename(image_path)
    ET.SubElement(root, 'path').text = "\\"+image_path
    size = ET.SubElement(root, 'size')
    ET.SubElement(size, 'width').text = str(image_width)
    ET.SubElement(size, 'height').text = str(image_height)
    ET.SubElement(size, 'depth').text = '3'

    # Convert the YOLO bounding boxes to XML format
    for line in lines:
        # Extract the class ID and bounding box coordinates from the YOLO annotation
        class_id, x_center, y_center, width, height = line.split()
        class_name = class_names[int(class_id)]
        x_min = int((float(x_center) - float(width) / 2) * image_width)
        y_min = int((float(y_center) - float(height) / 2) * image_height)
        x_max = int((float(x_center) + float(width) / 2) * image_width)
        y_max = int((float(y_center) + float(height) / 2) * image_height)

        # Create an XML object for the bounding box
        object_node = ET.SubElement(root, 'object')
        ET.SubElement(object_node, 'name').text = class_name
        ET.SubElement(object_node, 'pose').text = "Unspecified"
        ET.SubElement(object_node, 'truncated').text = '0'
        ET.SubElement(object_node, 'difficult').text = '0'

        bndbox = ET.SubElement(object_node, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(x_min)
        ET.SubElement(bndbox, 'ymin').text = str(y_min)
        ET.SubElement(bndbox, 'xmax').text = str(x_max)
        ET.SubElement(bndbox, 'ymax').text = str(y_max)

    # Write the XML file
    tree = ET.ElementTree(root)
    tree.write(xml_file_path)



# train data
folder_path = '\\data\\train\\labels'
img_path = '\\data\\complete\\images'
output_folder_path = '\\data\\complete\\images'
class_names = ['fish', 'jellyfish', 'penguin', 'puffin', 'shark', 'starfish', 'stingray']
for filename in glob.glob(os.path.join(folder_path, '*.txt')):
      with open(filename, 'r') as f:
        text = f.read()
        temp1=text.split('\n')
        file_name = filename.replace(folder_path,"")
        image_file = file_name.replace(".txt",'.jpg')
        yolo_annotation_path = folder_path+file_name
        image_folder_path = img_path+image_file
        convert_annotation(yolo_annotation_path, image_folder_path, output_folder_path, class_names)

# valid data
folder_path = '\\data\\valid\\labels'
img_path = '\\data\\complete\\images'
output_folder_path = '\\data\\complete\\images'
class_names = ['fish', 'jellyfish', 'penguin', 'puffin', 'shark', 'starfish', 'stingray']
for filename in glob.glob(os.path.join(folder_path, '*.txt')):
      with open(filename, 'r') as f:
        text = f.read()
        temp1=text.split('\n')
        file_name = filename.replace(folder_path,"")
        image_file = file_name.replace(".txt",'.jpg')
        yolo_annotation_path = folder_path+file_name
        image_folder_path = img_path+image_file
        convert_annotation(yolo_annotation_path, image_folder_path, output_folder_path, class_names)


# test data
folder_path = '\\data\\test\\labels'
img_path = '\\data\\complete\\images'
output_folder_path = '\\data\\complete\\images'
class_names = ['fish', 'jellyfish', 'penguin', 'puffin', 'shark', 'starfish', 'stingray']
for filename in glob.glob(os.path.join(folder_path, '*.txt')):
      with open(filename, 'r') as f:
        text = f.read()
        temp1=text.split('\n')
        file_name = filename.replace(folder_path,"")
        image_file = file_name.replace(".txt",'.jpg')
        yolo_annotation_path = folder_path+file_name
        image_folder_path = img_path+image_file
        convert_annotation(yolo_annotation_path, image_folder_path, output_folder_path, class_names)
