import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

def load_cifar10():
    """
    Memuat dataset CIFAR-10 dan melakukan preprocessing dasar.
    """
    # Memuat dataset
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    
    print("Ukuran data asli:")
    print(f"Training: {x_train.shape}, {y_train.shape}")
    print(f"Testing: {x_test.shape}, {y_test.shape}")
    
    # Normalisasi data
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # One-hot encoding untuk label
    y_train_categorical = to_categorical(y_train, 10)
    y_test_categorical = to_categorical(y_test, 10)
    
    return (x_train, y_train_categorical, y_train), (x_test, y_test_categorical, y_test)

def resize_data(x_train, x_test, target_size=(224, 224)):
    """
    Mengubah ukuran gambar untuk model pretrained.
    """
    # Membuat dataset kosong dengan ukuran baru
    x_train_resized = np.zeros((x_train.shape[0], target_size[0], target_size[1], 3), dtype=np.float32)
    x_test_resized = np.zeros((x_test.shape[0], target_size[0], target_size[1], 3), dtype=np.float32)
    
    # Mengubah ukuran setiap gambar
    for i in range(x_train.shape[0]):
        x_train_resized[i] = tf.image.resize(x_train[i], target_size).numpy()
    
    for i in range(x_test.shape[0]):
        x_test_resized[i] = tf.image.resize(x_test[i], target_size).numpy()
    
    print(f"Ukuran data setelah resize: {x_train_resized.shape}, {x_test_resized.shape}")
    
    return x_train_resized, x_test_resized

def create_data_generator(rotation_range=15, width_shift_range=0.1,
                          height_shift_range=0.1, horizontal_flip=True,
                          zoom_range=0.1):
    """
    Membuat generator data augmentasi.
    """
    train_datagen = ImageDataGenerator(
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        horizontal_flip=horizontal_flip,
        zoom_range=zoom_range
    )
    
    return train_datagen

def visualize_augmentation(x_train, y_train, train_datagen, num_samples=5):
    """
    Memvisualisasikan hasil augmentasi data.
    """
    # Pilih beberapa sampel acak
    indices = np.random.choice(range(len(x_train)), num_samples, replace=False)
    samples = x_train[indices]
    
    # Persiapkan generator untuk sampel ini
    aug_iter = train_datagen.flow(samples, y_train[indices], batch_size=num_samples)
    aug_images = next(aug_iter)
    
    # Visualisasikan perbandingan
    plt.figure(figsize=(2*num_samples, 4))
    
    # Gambar asli
    for i in range(num_samples):
        plt.subplot(2, num_samples, i+1)
        plt.imshow(samples[i])
        plt.title("Original")
        plt.axis('off')
    
    # Gambar hasil augmentasi
    for i in range(num_samples):
        plt.subplot(2, num_samples, num_samples+i+1)
        plt.imshow(aug_images[0][i])
        plt.title("Augmented")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('results/data_augmentation_examples.png')
    plt.show()

def preprocess_for_model(model_name, x_train, x_test):
    """
    Melakukan preprocessing khusus berdasarkan model yang akan digunakan.
    """
    if model_name.lower() == 'vgg16':
        # VGG16 preprocessing
        from tensorflow.keras.applications.vgg16 import preprocess_input
        x_train_processed = preprocess_input(x_train * 255.0)  # Undo normalization
        x_test_processed = preprocess_input(x_test * 255.0)
    
    elif model_name.lower() == 'resnet50':
        # ResNet50 preprocessing
        from tensorflow.keras.applications.resnet50 import preprocess_input
        x_train_processed = preprocess_input(x_train * 255.0)
        x_test_processed = preprocess_input(x_test * 255.0)
    
    elif model_name.lower() == 'mobilenetv2':
        # MobileNetV2 preprocessing
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        x_train_processed = preprocess_input(x_train * 255.0)
        x_test_processed = preprocess_input(x_test * 255.0)
    
    else:
        # Default: hanya gunakan data yang sudah dinormalisasi
        x_train_processed = x_train
        x_test_processed = x_test
    
    return x_train_processed, x_test_processed

def visualize_data_samples(x_train, y_train, num_samples=10):
    """
    Memvisualisasikan beberapa sampel dari dataset.
    """
    # Nama kelas CIFAR-10
    class_names = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
    
    # Pilih sampel acak
    indices = np.random.choice(range(len(x_train)), num_samples, replace=False)
    
    plt.figure(figsize=(15, 6))
    for i, idx in enumerate(indices):
        plt.subplot(2, 5, i+1)
        
        # Tampilkan gambar
        plt.imshow(x_train[idx])
        
        # Tampilkan label
        label = np.argmax(y_train[idx]) if len(y_train[idx].shape) > 0 else y_train[idx][0]
        plt.title(class_names[label])
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('results/data_samples.png')
    plt.show()

def create_vgg16_model_optimized(input_shape):
    if input_shape[0] == 32:  # Ukuran asli CIFAR-10
        model = models.Sequential([
            # Blok 1
            layers.Conv2D(64, (3, 3), padding='same', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(64, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            
            # Blok 2
            layers.Conv2D(128, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(128, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            
            # Blok 3
            layers.Conv2D(256, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(256, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            
            # Classifier - Diperkecil dari aslinya
            layers.Flatten(),
            layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
        ])
        
        # Juga perlu mengubah optimizer
        # model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
        #               loss='categorical_crossentropy',
        #               metrics=['accuracy'])
        
        return model

if __name__ == "__main__":
    # Contoh penggunaan
    import os
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # Memuat dataset
    print("Memuat dataset CIFAR-10...")
    (x_train, y_train_cat, y_train), (x_test, y_test_cat, y_test) = load_cifar10()
    
    # Visualisasi sampel
    print("\nMemvisualisasikan sampel dataset...")
    visualize_data_samples(x_train, y_train)
    
    # Membuat data generator
    print("\nMembuat data generator untuk augmentasi...")
    train_datagen = create_data_generator()
    
    # Visualisasi augmentasi
    print("\nMemvisualisasikan hasil augmentasi...")
    visualize_augmentation(x_train, y_train_cat, train_datagen)
    
    # Opsi 1: Gunakan dataset asli 32x32 (dengan adaptasi model)
    print("\nMenggunakan dataset asli 32x32...")
    
    # Opsi 2: Ubah ukuran dataset (untuk model pretrained)
    print("\nMengubah ukuran dataset ke 224x224...")
    x_train_resized, x_test_resized = resize_data(x_train, x_test)
    
    # Preprocessing untuk model tertentu
    print("\nMelakukan preprocessing untuk VGG16...")
    x_train_vgg, x_test_vgg = preprocess_for_model('vgg16', x_train_resized, x_test_resized)
    
    print("Preprocessing selesai.") 