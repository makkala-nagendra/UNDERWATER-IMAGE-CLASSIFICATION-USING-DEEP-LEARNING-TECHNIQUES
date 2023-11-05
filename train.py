from tflite_model_maker import object_detector
from tflite_model_maker import model_spec

# Datasets
train_data = object_detector.DataLoader.from_pascal_voc(
    './data/train/images',
    './data/train/images',
    ['fish', 'jellyfish', 'penguin', 'puffin', 'shark', 'starfish', 'stingray']
)
val_data = object_detector.DataLoader.from_pascal_voc(
    './data/valid/images',
    './data/valid/images',
    ['fish', 'jellyfish', 'penguin', 'puffin', 'shark', 'starfish', 'stingray']
)

# APIs for the model spec of TFLite Model Maker.
spec = model_spec.get('efficientdet_lite0')

# model
model = object_detector.create(train_data,
                               model_spec=spec,
                               batch_size=4,
                               train_whole_model=True,
                               epochs=40,
                               validation_data=val_data)

print(model.evaluate(val_data))

# save model
model.export(export_dir='./model', tflite_filename='model2.tflite')

print("training completed...")
