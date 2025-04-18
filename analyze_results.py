import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical
import seaborn as sns
import pandas as pd
import os
import pickle
import time
from sklearn.metrics import classification_report, confusion_matrix

# Nama kelas CIFAR-10
CIFAR_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

def load_cifar10():
    """Memuat dataset CIFAR-10."""
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    
    # Normalisasi data
    x_test = x_test.astype('float32') / 255.0
    
    # One-hot encoding untuk label
    y_test_categorical = to_categorical(y_test, 10)
    
    return x_test, y_test_categorical, y_test

def load_models():
    """Memuat model-model terlatih."""
    models = {}
    model_names = ['vgg16', 'resnet50', 'mobilenet']
    
    for name in model_names:
        model_path = f'models/{name}.h5'
        
        if os.path.exists(model_path):
            print(f"Memuat model {name}...")
            models[name] = load_model(model_path)
        else:
            print(f"Model {name} tidak ditemukan di {model_path}")
    
    return models

def evaluate_models(models, x_test, y_test_categorical):
    """Mengevaluasi kinerja model-model terlatih."""
    results = {}
    
    for name, model in models.items():
        print(f"\nEvaluasi model {name}...")
        
        # Evaluasi
        start_time = time.time()
        loss, accuracy = model.evaluate(x_test, y_test_categorical, verbose=0)
        eval_time = time.time() - start_time
        
        print(f"Akurasi: {accuracy:.4f}")
        print(f"Loss: {loss:.4f}")
        print(f"Waktu evaluasi: {eval_time:.2f} detik")
        
        # Prediksi
        start_time = time.time()
        y_pred = model.predict(x_test)
        inference_time = time.time() - start_time
        
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test_categorical, axis=1)
        
        # Classification report
        report = classification_report(y_true_classes, y_pred_classes, 
                                      target_names=CIFAR_CLASSES, output_dict=True)
        
        # Confusion Matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        
        # Inference time per image
        inference_time_per_image = inference_time / len(x_test)
        print(f"Waktu inferensi rata-rata per gambar: {inference_time_per_image*1000:.2f} ms")
        
        results[name] = {
            'accuracy': accuracy,
            'loss': loss,
            'eval_time': eval_time,
            'inference_time': inference_time,
            'inference_time_per_image': inference_time_per_image,
            'classification_report': report,
            'confusion_matrix': cm,
            'y_pred': y_pred
        }
    
    return results

def visualize_confusion_matrices(results):
    """Memvisualisasikan confusion matrix untuk setiap model."""
    for name, result in results.items():
        plt.figure(figsize=(10, 8))
        sns.heatmap(result['confusion_matrix'], annot=True, fmt='d', cmap='Blues', 
                   xticklabels=CIFAR_CLASSES, yticklabels=CIFAR_CLASSES)
        plt.title(f'Confusion Matrix - {name.upper()}')
        plt.ylabel('Label Sebenarnya')
        plt.xlabel('Label Prediksi')
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{name}_analysis.png')
        plt.show()

def compare_performance(results):
    """Membandingkan performa antar model."""
    model_names = list(results.keys())
    
    # Metrik perbandingan
    metrics = {
        'Model': model_names,
        'Akurasi': [results[name]['accuracy'] for name in model_names],
        'Loss': [results[name]['loss'] for name in model_names],
        'Waktu Evaluasi (s)': [results[name]['eval_time'] for name in model_names],
        'Waktu Inferensi per Gambar (ms)': [results[name]['inference_time_per_image'] * 1000 for name in model_names]
    }
    
    # Tambahkan metrik precision, recall, f1-score untuk setiap kelas
    for name in model_names:
        report = results[name]['classification_report']
        metrics[f'{name} Avg Precision'] = report['weighted avg']['precision']
        metrics[f'{name} Avg Recall'] = report['weighted avg']['recall']
        metrics[f'{name} Avg F1-Score'] = report['weighted avg']['f1-score']
    
    # Buat DataFrame
    df = pd.DataFrame(metrics)
    print("\nPerbandingan Performa Model:")
    print(df.to_string(index=False))
    
    # Simpan ke CSV
    df.to_csv('model_performance_comparison.csv', index=False)
    
    # Visualisasi perbandingan
    plt.figure(figsize=(14, 10))
    
    # Plot akurasi
    plt.subplot(2, 2, 1)
    sns.barplot(x='Model', y='Akurasi', data=df)
    plt.title('Akurasi')
    plt.ylim(0, 1)
    
    # Plot waktu inferensi
    plt.subplot(2, 2, 2)
    sns.barplot(x='Model', y='Waktu Inferensi per Gambar (ms)', data=df)
    plt.title('Waktu Inferensi per Gambar')
    
    # Plot precision, recall, f1-score
    plt.subplot(2, 2, 3)
    metrics_data = {
        'Model': [],
        'Metrik': [],
        'Nilai': []
    }
    
    for name in model_names:
        report = results[name]['classification_report']
        metrics_data['Model'].extend([name] * 3)
        metrics_data['Metrik'].extend(['Precision', 'Recall', 'F1-Score'])
        metrics_data['Nilai'].extend([
            report['weighted avg']['precision'],
            report['weighted avg']['recall'],
            report['weighted avg']['f1-score']
        ])
    
    metrics_df = pd.DataFrame(metrics_data)
    sns.barplot(x='Model', y='Nilai', hue='Metrik', data=metrics_df)
    plt.title('Metrik Klasifikasi')
    plt.ylim(0, 1)
    
    # Plot per-class accuracy
    plt.subplot(2, 2, 4)
    per_class_data = {
        'Model': [],
        'Class': [],
        'F1-Score': []
    }
    
    for name in model_names:
        report = results[name]['classification_report']
        for class_name in CIFAR_CLASSES:
            per_class_data['Model'].append(name)
            per_class_data['Class'].append(class_name)
            per_class_data['F1-Score'].append(report[class_name]['f1-score'])
    
    per_class_df = pd.DataFrame(per_class_data)
    sns.boxplot(x='Model', y='F1-Score', data=per_class_df)
    plt.title('Distribusi F1-Score per Model')
    
    plt.tight_layout()
    plt.savefig('performance_comparison_analysis.png')
    plt.show()
    
    # Heatmap untuk perbandingan per kelas
    plt.figure(figsize=(12, 8))
    per_class_pivot = per_class_df.pivot(index='Class', columns='Model', values='F1-Score')
    sns.heatmap(per_class_pivot, annot=True, cmap='YlGnBu', fmt=".3f")
    plt.title('F1-Score per Kelas untuk Setiap Model')
    plt.tight_layout()
    plt.savefig('per_class_comparison_analysis.png')
    plt.show()

def visualize_misclassifications(results, x_test, y_test):
    """Memvisualisasikan contoh kesalahan klasifikasi dari model terbaik."""
    # Temukan model dengan akurasi tertinggi
    best_model_name = max(results.keys(), key=lambda name: results[name]['accuracy'])
    best_model_result = results[best_model_name]
    
    print(f"\nMemvisualisasikan kesalahan klasifikasi dari model terbaik: {best_model_name}")
    
    y_pred = best_model_result['y_pred']
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.squeeze(y_test)
    
    # Temukan kesalahan klasifikasi
    misclassified_indices = np.where(y_pred_classes != y_true_classes)[0]
    
    if len(misclassified_indices) > 0:
        # Pilih 10 kesalahan klasifikasi secara acak
        sample_size = min(10, len(misclassified_indices))
        sample_indices = np.random.choice(misclassified_indices, sample_size, replace=False)
        
        plt.figure(figsize=(15, 8))
        for i, idx in enumerate(sample_indices):
            plt.subplot(2, 5, i+1)
            plt.imshow(x_test[idx])
            
            true_label = CIFAR_CLASSES[y_true_classes[idx]]
            pred_label = CIFAR_CLASSES[y_pred_classes[idx]]
            
            plt.title(f"Sebenarnya: {true_label}\nPrediksi: {pred_label}", color='red')
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('misclassification_examples.png')
        plt.show()
    else:
        print("Tidak ada kesalahan klasifikasi ditemukan.")

def main():
    # Memuat dataset
    print("Memuat dataset CIFAR-10...")
    x_test, y_test_categorical, y_test_original = load_cifar10()
    print(f"Ukuran data test: {x_test.shape}")
    
    # Memuat model-model terlatih
    models = load_models()
    
    if not models:
        print("Tidak ada model yang ditemukan. Jalankan cnn_comparison.py terlebih dahulu.")
        return
    
    # Evaluasi model
    results = evaluate_models(models, x_test, y_test_categorical)
    
    # Visualisasi confusion matrix
    visualize_confusion_matrices(results)
    
    # Bandingkan performa model
    compare_performance(results)
    
    # Visualisasi kesalahan klasifikasi
    visualize_misclassifications(results, x_test, y_test_original)
    
    print("\nAnalisis selesai. Hasil visualisasi telah disimpan.")

if __name__ == "__main__":
    main() 