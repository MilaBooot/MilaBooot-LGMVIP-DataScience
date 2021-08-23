import tensorflow as tf
import numpy as np
from tensorflow.keras import optimizers
from keras.datasets import mnist
import matplotlib.pyplot as plt
from PIL import Image

LR = 1e-4
dropout = 0.5


(x_train, y_train), (x_test, y_test) = mnist.load_data(path="mnist.npz")	##whyyy we split this?

#change the scale from 0-255 to 0-1
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

#Size and shape of the mnist dataset
print("x_train: ", str(x_train.shape)) 
print("y_train: ", str(y_train.shape)) 
print("x_test: ", str(x_test.shape))   
print("y_test: ", str(y_test.shape))

#now creat our model for training and testing 
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten())	#hidden layer 1 shoud be flat layer w/c is 28*28=784 nurons

model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
#compile our model
model.compile( loss= 'sparse_categorical_crossentropy',
			   optimizer=tf.keras.optimizers.Adam(learning_rate=LR),  
			   metrics=['accuracy'],
				)
val_loss, val_acc = model.evaluate(x_test, y_test)

model.fit(x_train, y_train, epochs=5, shuffle=True, validation_data=(x_test, y_test))
#save our trained model
model.save('final_model.model')
#to show images found @ datasets index number 3000
plt.imshow(x_train[3000], cmap = plt.cm.binary)
plt.show()

#to pridict numbers
new_model = tf.keras.models.load_model('final_model.model')
prediction = new_model.predict(x_test)
#print(prediction)

print(np.argmax(prediction[4]))	
plt.imshow(x_test[4])
plt.show()
plt.savefig('saved digit.jpg')
"""
#to predict new written number
img = np.invert(Image.open("timg.png").convert('L')).ravel()
with tf.Session() as sess:
	prediction = sess.run(tf.argmax(new_model, 1), feed_dict={X: [img]})
	print ("Prediction for test image:", np.squeeze(prediction))
"""