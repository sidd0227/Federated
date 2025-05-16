import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score

def create_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_shape,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_local_model(X_train, y_train, X_val, y_val, input_shape, num_classes):
    model = create_model(input_shape, num_classes)
    model.fit(X_train, y_train, epochs=5, batch_size=16, verbose=0, validation_data=(X_val, y_val))
    val_preds = np.argmax(model.predict(X_val), axis=1)
    val_acc = accuracy_score(y_val, val_preds)
    return model.get_weights(), val_acc
