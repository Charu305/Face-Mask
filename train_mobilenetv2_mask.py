import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

dataset_path = r"C:\Users\ASUS\Documents\Hope AI\Deep Learning\Week11-Deep Learning Module\Pytorch_Retinaface\dataset_cropped"

img_size = (224, 224)
epochs = 10
batch_size = 32

datagen = ImageDataGenerator(
    rescale=1.0/255,   # FIXED!!
    validation_split=0.2,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True)

train_data = datagen.flow_from_directory(
    dataset_path, target_size=img_size,
    batch_size=batch_size, class_mode='categorical', subset='training')

val_data = datagen.flow_from_directory(
    dataset_path, target_size=img_size,
    batch_size=batch_size, class_mode='categorical', subset='validation')

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in base_model.layers[:-20]:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
preds = Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=preds)

model.compile(optimizer=Adam(1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit(train_data, validation_data=val_data, epochs=epochs)

model.save("mask_classifier_mobilenetv2.h5")
print("\nModel saved as: mask_classifier_mobilenetv2.h5")
