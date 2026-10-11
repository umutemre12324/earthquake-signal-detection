# Deprem Erken Uyarı Sistemi

Bu proje, sentetik sismik sinyalleri **gürültü** veya **deprem sinyali** olarak sınıflandırmak için 1 boyutlu bir CNN modeli oluşturur. Veri üretimi, ön işleme, model eğitimi ve değerlendirme adımları ayrı Python dosyalarında yürütülür.

> **Önemli:** Proje sentetik verilerle hazırlanmış deneysel bir çalışmadır. Gerçek deprem algılama veya erken uyarı amacıyla kullanıma uygunluğu doğrulanmamıştır; can ve mal güvenliği kararlarında kullanılmamalıdır.

## Gereksinimler

- TensorFlow ile uyumlu bir Python sürümü
- NumPy
- Matplotlib
- scikit-learn
- PyTorch
- Seaborn

Bağımlılıkları kurmak için:

```bash
python -m pip install numpy tensorflow matplotlib scikit-learn torch seaborn
```

## Kurulum ve çalıştırma

Komutları proje kök dizininde çalıştırın.

### 1. Sentetik sinyalleri üretin

```bash
python preprocessing.py
```

Bu adım `data/` dizinine 1.000 örnek uzunluğunda sentetik deprem, gürültü ve gürültülü deprem sinyalleri kaydeder. Örnek sinyallerin grafiğini de ekranda gösterir.

### 2. Veriyi hazırlayın

```bash
python dataset.py
```

Deprem ve gürültü örnekleri eğitim ve test kümelerine ayırılır (%80 / %20), ölçeklendirilir ve aşağıdaki dosyalar `processed_data/` dizinine kaydedilir:

- `X_train.npy`, `y_train.npy`
- `X_test.npy`, `y_test.npy`

### 3. Modeli eğitin

```bash
python train.py
```

Eğitim, 30 epoch'a kadar sürer; doğrulama kaybına göre erken durdurma uygulanır. En iyi model `cnn1d_model.keras` olarak kaydedilir. Eğitim grafiği `results/training_history.png` dosyasına yazılır.

### 4. Modeli değerlendirin

```bash
python test.py
```

Test kaybı, doğruluk, sınıflandırma raporu ve karmaşıklık matrisi terminalde gösterilir. Karmaşıklık matrisi `confusion_matrix.png` olarak kaydedilir ve grafik penceresinde açılır.

`test.py` çalıştırılmadan önce `processed_data/` içindeki test dosyalarının ve `cnn1d_model.keras` model dosyasının mevcut olması gerekir. Model dosyası depoda bulunsa bile test verisini hazırlamak için 1. ve 2. adımların tamamlanması gerekir.

## Proje yapısı

```text
.
├── data/                  # Üretilen sentetik sinyaller
├── processed_data/        # Eğitim/test NumPy dizileri
├── results/               # Eğitim grafikleri
├── dataset.py             # Veriyi yükleme, bölme ve ölçeklendirme
├── model.py               # 1D CNN model tanımı
├── preprocessing.py       # Sentetik sinyal üretimi ve görselleştirme
├── train.py               # Model eğitimi
├── test.py                # Model değerlendirmesi
└── cnn1d_model.keras      # Eğitilmiş Keras modeli
```

## Sınıflar

| Etiket | Sınıf |
| --- | --- |
| `0` | Gürültü |
| `1` | Deprem sinyali |
