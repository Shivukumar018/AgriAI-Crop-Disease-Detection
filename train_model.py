import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

IMG_SIZE = 224
BATCH_SIZE = 32

# DATA AUGMENTATION
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=25,
    zoom_range=0.25,
    horizontal_flip=True,
    shear_range=0.2,
    brightness_range=[0.8,1.2]
)

# TRAINING DATA
train_data = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

# VALIDATION DATA
valid_data = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# PRINT CLASS INDICES
print("\nCLASS INDICES:")
print(train_data.class_indices)

# PRETRAINED MODEL
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)

# PARTIAL FINE-TUNING
base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

# FINAL MODEL
model = Sequential([
    base_model,

    GlobalAveragePooling2D(),

    Dense(128, activation='relu'),

    Dropout(0.5),

    Dense(4, activation='softmax')
])

# COMPILE MODEL
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# TRAIN MODEL
model.fit(
    train_data,
    validation_data=valid_data,
    epochs=5
)

# SAVE MODEL
model.save('models/crop_disease_model.h5')

print("\nMODEL TRAINED SUCCESSFULLY")