import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Flatten,MaxPool2D,Conv2D,Dropout,MaxPooling2D,BatchNormalization
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing import image
from keras import regularizers

train_path='./face_expression/train'
test_path='./face_expression/test'

# create objects for Data Generation
train_datagen = ImageDataGenerator(rescale=1/255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1/255)

# Augmenting train and test
train_set=train_datagen.flow_from_directory(train_path,
                                             target_size=(72,72),
                                             batch_size=32,
                                             color_mode='grayscale',
                                             class_mode='categorical')
test_set=test_datagen.flow_from_directory(test_path,
                                             target_size=(72,72),
                                             batch_size=32,
                                             color_mode='grayscale',
                                             class_mode='categorical')

train_set.class_indices

# See the shape of any data
train_sample=next(train_set)
print(train_sample[0].shape)

model=Sequential()

model.add(Conv2D(32, kernel_size=(2, 2), padding='same', activation='relu', input_shape =(72,72,1)))
model.add(Conv2D(64, kernel_size=(2, 2), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.30))

model.add(Conv2D(128, kernel_size=(2, 2), activation='relu', padding='same'))
model.add(Conv2D(256, kernel_size=(2, 2), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.30))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.30))

#model.add(Dense(256, activation='tanh'))
#model.add(Dropout(0.25))

#model.add(Dense(128, activation='relu'))
#model.add(Dropout(0.25))

    
model.add(Dense(7, activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.summary()

early=EarlyStopping(monitor='val_accuracy',patience=5,restore_best_weights=True,verbose=1,min_delta=0.001)

history=model.fit(train_set,epochs=100,validation_data=test_set,batch_size=32)

# we get the dictionary 'history'
history.history.keys()

fig,ax=plt.subplots(1,2,figsize=(10,5))

ax[0].plot(history.history['loss'],label='training')
ax[0].plot(history.history['val_loss'],label='validation')
ax[0].legend()
ax[0].set_title('Loss with epochs')

ax[1].plot(history.history['accuracy'],label='training')
ax[1].plot(history.history['val_accuracy'],label='validation')
ax[1].legend()
ax[1].set_title('Accuracy with epochs')

model.save('fer_abhigyan.h5')

img_path='./face_expression/train/angry/Training_3908.jpg'
test_image=image.load_img(img_path,target_size=(72,72),color_mode='grayscale')

type(test_image)
test_image=image.img_to_array(test_image)

print(test_image.shape)
plt.imshow(test_image)

test_image=test_image.reshape(1,72,72,1)

classes=['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']
result=model.predict(test_image)

result[0]

y_pred=np.argmax(result[0])
print('The person is ',classes[y_pred])
