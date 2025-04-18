import os
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description='Jalankan eksperimen perbandingan CNN pada CIFAR-10')
    parser.add_argument('--original_size', action='store_true', help='Gunakan ukuran asli 32x32 (tanpa resize)')
    parser.add_argument('--only_analysis', action='store_true', help='Hanya jalankan analisis hasil tanpa melatih model')
    parser.add_argument('--epochs', type=int, default=20, help='Jumlah epoch untuk pelatihan')
    args = parser.parse_args()
    
    # Buat direktori untuk model dan hasil
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Tampilkan informasi eksperimen
    print("=" * 50)
    print(f"EKSPERIMEN PERBANDINGAN ARSITEKTUR CNN PADA CIFAR-10")
    print("=" * 50)
    print(f"Menggunakan ukuran asli (32x32): {args.original_size}")
    if not args.only_analysis:
        print(f"Jumlah epoch: {args.epochs}")
    print("=" * 50)
    
    if not args.only_analysis:
        # Jalankan preprocessing dataset
        print("\n1. PREPROCESSING DATASET")
        print("-" * 50)
        subprocess.run(['python', 'preprocess.py'])
        
        # Edit file cnn_comparison.py untuk menyesuaikan parameter
        with open('cnn_comparison.py', 'r') as file:
            lines = file.readlines()
        
        for i, line in enumerate(lines):
            # Set ukuran input
            if 'use_original_size = ' in line:
                lines[i] = f"    use_original_size = {str(args.original_size)}  # Ubah ke False untuk menggunakan ukuran 224x224\n"
            # Set jumlah epoch
            if "def train_model(model, model_name, x_train, y_train, x_test, y_test, epochs=" in line:
                lines[i] = f"def train_model(model, model_name, x_train, y_train, x_test, y_test, epochs={args.epochs}, batch_size=64):\n"
        
        with open('cnn_comparison.py', 'w') as file:
            file.writelines(lines)
        
        # Jalankan pelatihan dan evaluasi model
        print("\n2. PELATIHAN DAN EVALUASI MODEL")
        print("-" * 50)
        subprocess.run(['python', 'cnn_comparison.py'])
    
    # Jalankan analisis hasil
    print("\n3. ANALISIS HASIL")
    print("-" * 50)
    subprocess.run(['python', 'analyze_results.py'])
    
    print("\n" + "=" * 50)
    print("EKSPERIMEN SELESAI")
    print("=" * 50)
    print("\nHasil telah disimpan di direktori 'results/'")
    print("Model terlatih telah disimpan di direktori 'models/'")

if __name__ == "__main__":
    main() 