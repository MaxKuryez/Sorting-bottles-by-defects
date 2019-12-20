import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.utils import plot_model
import matplotlib.pyplot as plt

datagen = ImageDataGenerator()

#loading the training and testing data from directory
train_it = datagen.flow_from_directory('C://Users/HP/Desktop/DeepLearning/new_bottles/train/', class_mode='binary',target_size=(28,28),batch_size=40,color_mode='grayscale')
test_it = datagen.flow_from_directory('C://Users/HP/Desktop/DeepLearning/new_bottles/test/', class_mode='binary',target_size=(28,28),batch_size=60,color_mode='grayscale')

#creating the training and testing arrays
train_images, train_labels = train_it.next()
test_images, test_labels = test_it.next()

#image normalization
train_images = (train_images / 255) - 0.5
test_images = (test_images / 255) - 0.5

#reshaping the array in order to get 1-dimentional array for training
train_images = train_images.flatten().reshape(40, 784)
test_images = test_images.flatten().reshape(60, 784)

#initializing the model and its layer structure
model = Sequential([
  Dense(64, activation='relu', input_shape=(784,)),
  Dense(64, activation='relu'),
  Dense(8, activation='sigmoid'),
])

#configuring the model for training
model.compile(
  optimizer='adam',
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)

#actual training of the model + saving the training history for graphs
history = model.fit(
  train_images,
  to_categorical(train_labels),
  epochs=300,
  batch_size=70,
)

#evaluation of the weights
model.evaluate(
  test_images,
  to_categorical(test_labels)
)

#saving the weights
model.save_weights('model.h5')
#saving the graph representation of network structure
#plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

#creating the predictions array
predictions = model.predict(test_images)
test_labels = test_labels.astype(int)

print(np.argmax(predictions, axis=1))
print(test_labels)

#creating and saving to file the accuracy plot
plt.plot(history.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('accuracy.png')
plt.show()
#creating and saving to file the loss plot
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('loss.png')