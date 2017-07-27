from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
#from keras.applications import InceptionV3
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import model_from_json
import numpy as np

img_width, img_height = 150, 150
validation_data_dir = 'data/test'
nb_validation_samples = 1
epochs = 20
batch_size = 1

json_file = open('first_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("first_try.h5")
print("Loaded model from disk")
 
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

test_datagen = ImageDataGenerator(rescale=1. / 255)

validation_generator = test_datagen.flow_from_directory(
	validation_data_dir,
	target_size=(img_width, img_height),
	batch_size=batch_size,
	class_mode='binary')

features = loaded_model.predict_generator(
	generator=validation_generator,
	steps=nb_validation_samples // batch_size,
	max_queue_size=10)

if features > 0.5:
	print("############# Its a dog #############")
else:
	print("############# Its a cat #############")
