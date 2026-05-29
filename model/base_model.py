import mediapipe as mp
import numpy as np
import tensorflow as tf
import os 
from sklearn.model_selection import train_test_split    
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd


numpy_files = "C:\\Users\\asus\\Downloads\\FingerHint\\numpy__dataset"
X = np.load(os.path.join(numpy_files, "X_features.npy"))
y = np.load(os.path.join(numpy_files, "y_labels.npy"))
Class = np.load(os.path.join(numpy_files, "classes.npy"))

print(X.shape)
print(y.shape)
print(Class.shape)


X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(63,)),
    tf.keras.layers.Dropout(0.3), # ปิดการทำงานนิวรอนแบบสุ่ม 30% กัน Overfitting
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(24, activation='softmax') # 
]) #MLP mutil layer perceptron


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

history = model.fit(X_train, y_train,
                    epochs=15,          
                    batch_size=32,      
                    validation_data=(X_val, y_val))

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc:.4f}')
print(classification_report(y_test, model.predict(X_test).argmax(axis=1), target_names=Class))
smooth_table = confusion_matrix(y_test, model.predict(X_test).argmax(axis=1))

df = pd.DataFrame(smooth_table, index=Class, columns=Class)
print(df)
#model.save("finger_hint_model_newone_data.keras")
#print("โมเดลถูกบันทึกเรียบร้อยแล้วที่ finger_hint_model_newone_data.keras")

