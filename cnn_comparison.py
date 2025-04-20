import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import time
import pandas as pd
import os
from preprocess import load_cifar10, resize_data, create_data_generator, preprocess_for_model, create_vgg16_model_optimized

# Set random seed untuk reprodusibilitas
np.random.seed(42)
tf.random.set_seed(42)

print("TensorFlow versi:", tf.__version__)
print("GPU tersedia:", len(tf.config.list_physical_devices('GPU')))
print("Nama GPU:", tf.test.gpu_device_name())

# Fungsi untuk membuat model VGG16
def create_vgg16_model(input_shape):
    # Gunakan model yang dioptimasi untuk ukuran 32x32
    if input_shape[0] == 32:
        return create_vgg16_model_optimized(input_shape)
    else:
        # Kode existing untuk ukuran 224x224
        # Gunakan model pretrained
        base_model = applications.VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        
        # Freeze base model
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
        ])
    
    return model

# Fungsi untuk membuat model ResNet50
def create_resnet50_model(input_shape):
    # Dua opsi, tergantung ukuran input
    if input_shape[0] == 32:  # Ukuran asli CIFAR-10
        # Buat model dengan arsitektur ResNet yang lebih kecil
        def residual_block(x, filters, kernel_size=3, stride=1, conv_shortcut=False):
            shortcut = x
            
            if conv_shortcut:
                shortcut = layers.Conv2D(filters, 1, strides=stride)(shortcut)
                shortcut = layers.BatchNormalization()(shortcut)
            
            x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            
            x = layers.Conv2D(filters, kernel_size, padding='same')(x)
            x = layers.BatchNormalization()(x)
            
            x = layers.add([x, shortcut])
            x = layers.Activation('relu')(x)
            
            return x
        
        # Input
        inputs = layers.Input(shape=input_shape)
        
        # Initial convolution
        x = layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
        
        # Residual blocks
        x = residual_block(x, 64, conv_shortcut=True)
        x = residual_block(x, 64)
        
        x = residual_block(x, 128, stride=2, conv_shortcut=True)
        x = residual_block(x, 128)
        
        x = residual_block(x, 256, stride=2, conv_shortcut=True)
        x = residual_block(x, 256)
        
        # Global pooling and classifier
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(10, activation='softmax')(x)
        
        model = models.Model(inputs, x)
    else:  # Ukuran setelah resize (misalnya 224x224)
        # Gunakan model pretrained
        base_model = applications.ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        
        # Freeze base model
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
        ])
    
    return model

# Fungsi untuk membuat model MobileNetV2
def create_mobilenet_model(input_shape):
    # Dua opsi, tergantung ukuran input
    if input_shape[0] == 32:  # Ukuran asli CIFAR-10
        # Buat model dengan arsitektur MobileNet yang lebih kecil
        
        def inverted_residual_block(x, expand_ratio, filters, stride):
            # Input
            input_tensor = x
            in_channels = input_tensor.shape[-1]
            
            # Expansion phase
            x = layers.Conv2D(expand_ratio * in_channels, kernel_size=1, padding='same', use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU(6.)(x)
            
            # Depthwise convolution
            x = layers.DepthwiseConv2D(
                kernel_size=3, strides=stride, padding='same', use_bias=False
            )(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU(6.)(x)
            
            # Projection
            x = layers.Conv2D(filters, kernel_size=1, padding='same', use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            
            # Skip connection
            if stride == 1 and in_channels == filters:
                x = layers.add([input_tensor, x])
            
            return x
        
        # Input
        inputs = layers.Input(shape=input_shape)
        
        # Initial convolution
        x = layers.Conv2D(32, kernel_size=3, strides=2, padding='same', use_bias=False)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU(6.)(x)
        
        # Inverted residual blocks
        x = inverted_residual_block(x, expand_ratio=1, filters=16, stride=1)
        
        x = inverted_residual_block(x, expand_ratio=6, filters=24, stride=2)
        x = inverted_residual_block(x, expand_ratio=6, filters=24, stride=1)
        
        x = inverted_residual_block(x, expand_ratio=6, filters=32, stride=2)
        x = inverted_residual_block(x, expand_ratio=6, filters=32, stride=1)
        
        x = inverted_residual_block(x, expand_ratio=6, filters=64, stride=2)
        x = inverted_residual_block(x, expand_ratio=6, filters=64, stride=1)
        
        # Global pooling and classifier
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(10, activation='softmax')(x)
        
        model = models.Model(inputs, x)
    else:  # Ukuran setelah resize (misalnya 224x224)
        # Gunakan model pretrained
        base_model = applications.MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        
        # Freeze base model
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
        ])
    
    return model

# Fungsi untuk melatih model
def train_model(model, model_name, x_train, y_train, x_test, y_test, epochs=5, batch_size=64):
    if model_name == 'vgg16':
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
    else:
        optimizer = 'adam'
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        f'models/{model_name}.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
    
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True
    )
    
    # Data augmentasi
    datagen = create_data_generator()
    datagen.fit(x_train)
    
    # Mencatat waktu mulai pelatihan
    start_time = time.time()
    
    # Pelatihan model
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=batch_size),
        epochs=epochs,
        validation_data=(x_test, y_test),
        callbacks=[checkpoint, early_stopping]
    )
    
    # Mencatat waktu selesai pelatihan
    training_time = time.time() - start_time
    print(f"Waktu pelatihan untuk {model_name}: {training_time:.2f} detik")
    
    # Evaluasi model
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test accuracy for {model_name}: {test_acc:.4f}")
    
    # Prediksi
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Classification report
    report = classification_report(y_true_classes, y_pred_classes, target_names=CIFAR_CLASSES, output_dict=True)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    
    # Inference time (rata-rata waktu per gambar)
    start_time = time.time()
    model.predict(x_test[:100])
    inference_time = (time.time() - start_time) / 100
    print(f"Waktu inferensi rata-rata per gambar untuk {model_name}: {inference_time*1000:.2f} ms")
    
    return {
        'history': history,
        'test_accuracy': test_acc,
        'classification_report': report,
        'confusion_matrix': cm,
        'training_time': training_time,
        'inference_time': inference_time
    }

# Fungsi untuk memvisualisasikan hasil pelatihan
def visualize_training_history(histories, model_names):
    plt.figure(figsize=(12, 5))
    
    # Plot akurasi
    plt.subplot(1, 2, 1)
    for i, model_name in enumerate(model_names):
        plt.plot(histories[i]['history'].history['accuracy'], label=f'{model_name} (train)')
        plt.plot(histories[i]['history'].history['val_accuracy'], label=f'{model_name} (val)')
    
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    for i, model_name in enumerate(model_names):
        plt.plot(histories[i]['history'].history['loss'], label=f'{model_name} (train)')
        plt.plot(histories[i]['history'].history['val_loss'], label=f'{model_name} (val)')
    
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/training_history.png')
    plt.show()

# Fungsi untuk memvisualisasikan confusion matrix
def visualize_confusion_matrix(cm, model_name):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CIFAR_CLASSES, yticklabels=CIFAR_CLASSES)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'results/confusion_matrix_{model_name}.png')
    plt.show()

# Fungsi untuk visualisasi prediksi
def visualize_predictions(model, x_test, y_test, num_images=10):
    # Prediksi
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Pilih sampel acak
    indices = np.random.choice(range(len(x_test)), num_images, replace=False)
    
    plt.figure(figsize=(15, 8))
    for i, idx in enumerate(indices):
        plt.subplot(2, 5, i+1)
        plt.imshow(x_test[idx])
        
        true_label = CIFAR_CLASSES[y_true_classes[idx]]
        pred_label = CIFAR_CLASSES[y_pred_classes[idx]]
        
        color = 'green' if true_label == pred_label else 'red'
        
        plt.title(f"True: {true_label}\nPred: {pred_label}", color=color)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('results/predictions_visualization.png')
    plt.show()

# Fungsi untuk membandingkan metrik
def compare_metrics(results, model_names):
    metrics = {
        'Model': model_names,
        'Test Accuracy': [results[i]['test_accuracy'] for i in range(len(model_names))],
        'Training Time (s)': [results[i]['training_time'] for i in range(len(model_names))],
        'Inference Time (ms)': [results[i]['inference_time'] * 1000 for i in range(len(model_names))]
    }
    
    # Tambahkan metrik precision, recall, f1-score untuk setiap kelas
    for i, model_name in enumerate(model_names):
        report = results[i]['classification_report']
        metrics[f'{model_name} Avg Precision'] = report['weighted avg']['precision']
        metrics[f'{model_name} Avg Recall'] = report['weighted avg']['recall']
        metrics[f'{model_name} Avg F1-Score'] = report['weighted avg']['f1-score']
    
    df = pd.DataFrame(metrics)
    print(df)
    
    # Visualisasi perbandingan
    plt.figure(figsize=(14, 10))
    
    # Plot akurasi
    plt.subplot(2, 2, 1)
    plt.bar(model_names, [results[i]['test_accuracy'] for i in range(len(model_names))])
    plt.title('Test Accuracy')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    
    # Plot waktu pelatihan
    plt.subplot(2, 2, 2)
    plt.bar(model_names, [results[i]['training_time'] for i in range(len(model_names))])
    plt.title('Training Time')
    plt.ylabel('Time (s)')
    
    # Plot waktu inferensi
    plt.subplot(2, 2, 3)
    plt.bar(model_names, [results[i]['inference_time'] * 1000 for i in range(len(model_names))])
    plt.title('Inference Time')
    plt.ylabel('Time per image (ms)')
    
    # Plot F1-score
    plt.subplot(2, 2, 4)
    plt.bar(model_names, [results[i]['classification_report']['weighted avg']['f1-score'] for i in range(len(model_names))])
    plt.title('Weighted F1-Score')
    plt.ylabel('F1-Score')
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('results/metrics_comparison.png')
    plt.show()

# Main function
def main():
    # Buat direktori untuk menyimpan model dan hasil
    if not os.path.exists('models'):
        os.makedirs('models')
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # Pilih metode persiapan data
    use_original_size = True  # Ubah ke False untuk menggunakan ukuran 224x224
    
    # Load dan preprocess dataset
    print("Memuat dataset CIFAR-10...")
    (x_train, y_train_cat, y_train_orig), (x_test, y_test_cat, y_test_orig) = load_cifar10()
    
    if use_original_size:
        # Gunakan ukuran asli 32x32
        print("Menggunakan ukuran asli 32x32...")
        input_shape = (32, 32, 3)
        x_train_processed, x_test_processed = x_train, x_test
        y_train, y_test = y_train_cat, y_test_cat
    else:
        # Resize ke 224x224 untuk model pretrained
        print("Mengubah ukuran gambar ke 224x224...")
        x_train_resized, x_test_resized = resize_data(x_train, x_test)
        input_shape = (224, 224, 3)
        
        # Tidak ada preprocessing khusus saat ini
        x_train_processed, x_test_processed = x_train_resized, x_test_resized
        y_train, y_test = y_train_cat, y_test_cat
    
    # Buat dan latih model
    model_names = ['VGG16', 'ResNet50', 'MobileNetV2']
    model_results = []
    
    # VGG16
    print("\nMemulai pelatihan VGG16...")
    vgg_model = create_vgg16_model(input_shape)
    vgg_results = train_model(vgg_model, 'vgg16', x_train_processed, y_train, x_test_processed, y_test, epochs=5)
    model_results.append(vgg_results)
    
    # ResNet50
    print("\nMemulai pelatihan ResNet50...")
    resnet_model = create_resnet50_model(input_shape)
    resnet_results = train_model(resnet_model, 'resnet50', x_train_processed, y_train, x_test_processed, y_test)
    model_results.append(resnet_results)
    
    # MobileNetV2
    print("\nMemulai pelatihan MobileNetV2...")
    mobilenet_model = create_mobilenet_model(input_shape)
    mobilenet_results = train_model(mobilenet_model, 'mobilenet', x_train_processed, y_train, x_test_processed, y_test)
    model_results.append(mobilenet_results)
    
    # Visualisasi hasil pelatihan
    visualize_training_history(model_results, model_names)
    
    # Visualisasi confusion matrix untuk setiap model
    for i, model_name in enumerate(model_names):
        visualize_confusion_matrix(model_results[i]['confusion_matrix'], model_name)
    
    # Visualisasi prediksi dari model terbaik
    best_model_idx = np.argmax([results['test_accuracy'] for results in model_results])
    best_model_name = model_names[best_model_idx]
    
    if best_model_name == 'VGG16':
        best_model = vgg_model
    elif best_model_name == 'ResNet50':
        best_model = resnet_model
    else:
        best_model = mobilenet_model
    
    print(f"\nModel terbaik: {best_model_name}")
    visualize_predictions(best_model, x_test_processed, y_test)
    
    # Perbandingan metrik
    compare_metrics(model_results, model_names)
    
    # Simpan hasil perbandingan ke file CSV
    metrics_df = pd.DataFrame({
        'Model': model_names,
        'Accuracy': [results['test_accuracy'] for results in model_results],
        'Training_Time': [results['training_time'] for results in model_results],
        'Inference_Time': [results['inference_time'] * 1000 for results in model_results],
        'Precision': [results['classification_report']['weighted avg']['precision'] for results in model_results],
        'Recall': [results['classification_report']['weighted avg']['recall'] for results in model_results],
        'F1_Score': [results['classification_report']['weighted avg']['f1-score'] for results in model_results]
    })
    
    metrics_df.to_csv('results/model_comparison.csv', index=False)
    print("\nHasil perbandingan telah disimpan ke 'results/model_comparison.csv'")

# Nama kelas CIFAR-10
CIFAR_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

if __name__ == "__main__":
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        print("GPU ditemukan! Mengaktifkan GPU...")
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    else:
        print("PERINGATAN: Tidak menemukan GPU, menggunakan CPU.")
    main() 