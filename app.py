
import os
import tensorflow as tf
from azureml.core import Run
from azureml.core.model import Model
from tensorflow.keras import datasets, layers, models

# 1. Configuration
batch_size = 32
epochs = 5
learning_rate = 0.001

# 2. Get Current Run
run = Run.get_context()

# 3. Load and Preprocess
(X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()
X_train, X_test = X_train/255.0, X_test/255.0

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# 4. Create a CNN Model
class CNNModel(tf.keras.Model):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = layers.Conv2D(32, (3,3), activation = 'relu', input_shape(28,28,1))
        self.maxpool1 = layers.MaxPooling2D((2,2))
        self.conv2 = layers.Conv2D(64, (3,3), activation = 'relu')
        self.maxpool2 = layers.MaxPooling2D((2,2))
        self.conv3 = layers.Conv2D(64,(3,3), activation = 'relu')
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(64, activation = 'relu')
        self.fc2 = layers.Dense(10)
    
    def call(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return self.fc2(x)
model = CNNModel()
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate),
                loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
                metrics = ['accuracy']
)

# 5. Train the Model
history = model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size, validation_data = (X_test, y_test))

# 6. Save the model as Tensorflow SavedModel
os.makedirs("outputs",exist_ok=True)
tf.saved_model.save(model, "outputs/mnist_cnn_model")

# 7. Log Metrics and Artifacts
for epoch in range(epochs):
    run.log('epoch', epoch)
    run.log('loss', history.history['loss'][epoch])
    run.log('accuracy', history.history['accuracy'][epoch])   
    run.log('val_loss', history.history['val_loss'][epoch])   
    run.log('val_accuracy', history.history['val_accuracy'][epoch])  

# 8. Complete Run

run.complete()        
