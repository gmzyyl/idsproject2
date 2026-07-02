# bitirme.py
# CIC-IDS2017 IDS Projesi - ML + DL
# Gamze Yaylı

import glob

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE 
import time

# -----------------------
# 1️⃣ VERİYİ OKU 
# -----------------------
print("Tüm veriler okunuyor (Birkaç saniye sürebilir)...")
all_files = glob.glob("*.csv") 
df_list = []

for file in all_files:
    temp = pd.read_csv(file)
    temp.columns = temp.columns.str.strip()
    df_list.append(temp)

df = pd.concat(df_list, ignore_index=True)
print("Orijinal Veri boyutu:", df.shape)

# -----------------------
# 2️⃣ VERİ TEMİZLEME VE FİLTRELEME
# -----------------------
print("\nVeri temizleniyor...")

# 1. Sonsuz değerleri NaN yap
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# 2. Tamamen boş olan sütunları at
df.dropna(axis=1, how='all', inplace=True)

# 3. SAYISAL sütunları 0 ile doldur
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(0)

# 4. METİN/STRING sütunları "0" (metin olarak) ile doldur
string_cols = df.select_dtypes(exclude=[np.number]).columns
df[string_cols] = df[string_cols].fillna("0")

# 5. Label sütununda NaN kalmışsa satırı düşür
df.dropna(subset=['Label'], inplace=True)

# 6. KRİTİK ADIM: LabelEncoder'ı burada çalıştırıp X ve y'yi tanımlıyoruz
le = LabelEncoder()
df['Label'] = le.fit_transform(df['Label'])

X = df.drop('Label', axis=1).values
y = df['Label'].values # İşte bilgisayar artık 'y'nin ne olduğunu burada anlıyor!


# -----------------------
# 3️⃣ AKILLI DENGELEME (UnderSampling + SMOTE)
# -----------------------
hedef_sayi = 3000 # Her sınıf için tam 3000 örnek istiyoruz!
print(f"\nVeriler her sınıf için {hedef_sayi} olacak şekilde dengeleniyor...")

# ADIM A: Çok olanları (Örn: 2 milyonluk BENIGN) 3000'e kırp
kucultulecekler = {label: hedef_sayi for label, count in zip(*np.unique(y, return_counts=True)) if count > hedef_sayi}
rus = RandomUnderSampler(sampling_strategy=kucultulecekler, random_state=42)
X_rus, y_rus = rus.fit_resample(X, y)

# ADIM B: Az olanları (Örn: 11 tane Heartbleed) SMOTE ile 3000'e çoğalt
buyutulecekler = {label: hedef_sayi for label, count in zip(*np.unique(y_rus, return_counts=True)) if count < hedef_sayi}
# Komşu sayısını (k_neighbors) dinamik ayarlıyoruz ki nadir sınıflarda hata vermesin
smote = SMOTE(sampling_strategy=buyutulecekler, random_state=42, k_neighbors=min(5, min(np.unique(y_rus, return_counts=True)[1])-1))
X_resampled, y_resampled = smote.fit_resample(X_rus, y_rus)

# Dengeleme sonrası durumu görelim
unique, counts = np.unique(y_resampled, return_counts=True)
print("\n🎉 Mükemmel Dengelenmiş Yeni Sınıf Dağılımı:\n", dict(zip(le.inverse_transform(unique), counts)))
print(f"Yeni Veri Seti Boyutu: {X_resampled.shape}")

# -----------------------
# 4️⃣ ÖZELLİKLERİ ÖLÇEKLEME VE VERİYİ BÖLME
# -----------------------
# Makine öğrenmesi modelleri için veriyi 0 ortalama ve 1 varyans olacak şekilde ölçekliyoruz
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resampled)

# Veriyi %80 Eğitim, %20 Test olacak şekilde ayırıyoruz
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)
print("\nEğitim boyutu:", X_train.shape, "Test boyutu:", X_test.shape)


# -----------------------
# 5️⃣ SKLEARN MODELLERİ
# -----------------------
ml_models = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'DecisionTree': DecisionTreeClassifier(),
    'RandomForest': RandomForestClassifier(),
    'SVM': SVC(),
    'KNN': KNeighborsClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'GradientBoosting': GradientBoostingClassifier()
}

print("\n--- Sklearn Modelleri ---")
for name, model in ml_models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n{name} performans:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average='macro', zero_division=0))
    print("Recall:", recall_score(y_test, y_pred, average='macro', zero_division=0))
    print("F1:", f1_score(y_test, y_pred, average='macro', zero_division=0))
    
    # ✅ Sklearn modelini kaydet
    joblib.dump(model, f"{name}.joblib")
    print(f"{name} kaydedildi!")

# -----------------------
# 6️⃣ PYTORCH DERİN ÖĞRENME MODELLERİ (ANN, CNN, LSTM)
# -----------------------
print("\n--- PyTorch Derin Öğrenme Modelleri ---")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Kullanılan Cihaz: {device}")

# Verileri tensöre çevirme
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

# ------------------------
# MODEL TANIMLARI
# ------------------------
# --- 1. MODEL: ANN (Yapay Sinir Ağı) ---
class ANN_Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, activation_fn):
        super(ANN_Model, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout = nn.Dropout(0.3)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)
        self.act = activation_fn
        
    def forward(self, x):
        x = self.act(self.bn1(self.fc1(x)))
        x = self.dropout(self.act(self.fc2(x)))
        x = self.fc3(x)
        return x

# --- 2. MODEL: 1D CNN (Evrişimli Sinir Ağı) ---
class CNN_Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, activation_fn):
        super(CNN_Model, self).__init__()
        # Giriş: (batch_size, 1, input_size)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # Flatten sonrası boyut hesabı (input_size / 2 pooling'den dolayı)
        linear_input = 64 * (input_size // 2) 
        self.fc1 = nn.Linear(linear_input, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.act = activation_fn
        
    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.act(self.conv2(x))
        x = x.view(x.size(0), -1) # Flatten (Düzleştirme)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x

# --- 3. MODEL: LSTM (Uzun-Kısa Süreli Bellek) ---
class LSTM_Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, activation_fn):
        super(LSTM_Model, self).__init__()
        # LSTM katmanında aktivasyon sabittir, ancak FC katmanında senin seçtiğini kullanırız
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=2, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, num_classes)
        self.act = activation_fn
        
    def forward(self, x):
        # x şekli: (batch_size, sequence_length, features)
        out, _ = self.lstm(x)
        # Sadece son zaman adımını (last sequence) al
        out = out[:, -1, :] 
        out = self.act(self.fc1(out))
        out = self.fc2(out)
        return out

# --- DEĞERLENDİRME FONKSİYONU ---
def evaluate_pytorch_model(model, X_test_tensor, y_test_tensor, model_name):
    model.eval()
    with torch.no_grad():
        # CNN ve LSTM için boyutu (batch_size, 1, features) yap
        if model_name in ['CNN', 'LSTM']:
            X_test_tensor = X_test_tensor.unsqueeze(1)
            
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs, 1)
        
        y_true = y_test_tensor.cpu().numpy()
        y_pred = predicted.cpu().numpy()
        
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        return acc, prec, rec, f1

# --- GRID SEARCH VE EĞİTİM DÖNGÜSÜ ---
input_dim = X_train.shape[1]
num_classes = len(np.unique(y))

# Denenecek Hiperparametreler (Proje Formuna Uygun)
activations = {'ReLU': nn.ReLU(), 'Tanh': nn.Tanh(), 'LeakyReLU': nn.LeakyReLU()}
learning_rates = [0.001, 0.01]
epochs = 10
batch_size = 256

model_classes = {'ANN': ANN_Model, 'CNN': CNN_Model, 'LSTM': LSTM_Model}
best_models = {}

print("\n--- Derin Öğrenme Grid Search Başlıyor ---")
for model_name, ModelClass in model_classes.items():
    best_acc = 0
    best_params = {}
    
    print(f"\n[{model_name}] İçin Hiperparametre Optimizasyonu:")
    
    for act_name, act_fn in activations.items():
        for lr in learning_rates:
            print(f"  Deney: Aktivasyon={act_name}, LR={lr}...")
            
            # Modeli başlat
            model = ModelClass(input_dim, 64, num_classes, act_fn).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)
            
            # Eğitim Döngüsü
            model.train()
            for epoch in range(epochs):
                permutation = torch.randperm(X_train_tensor.size(0))
                for i in range(0, X_train_tensor.size(0), batch_size):
                    indices = permutation[i:i + batch_size]
                    batch_x, batch_y = X_train_tensor[indices], y_train_tensor[indices]
                    
                    # CNN ve LSTM için (Batch, Channel/Seq, Features) formatına çevir
                    if model_name in ['CNN', 'LSTM']:
                        batch_x = batch_x.unsqueeze(1)
                    
                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
            
            # Değerlendirme
            acc, prec, rec, f1 = evaluate_pytorch_model(model, X_test_tensor, y_test_tensor, model_name)
            
            if acc > best_acc:
                best_acc = acc
                best_params = {'Aktivasyon': act_name, 'LR': lr, 'Model': model}
                
    # En iyi modeli kaydet
    print(f"\n⭐ {model_name} İçin En İyi Sonuçlar:")
    print(f"Parametreler: {best_params['Aktivasyon']}, LR={best_params['LR']}")
    print(f"Accuracy: {best_acc:.4f}")
    
    best_model = best_params['Model']
    torch.save(best_model.state_dict(), f"{model_name}_best.pth")
    print(f"✔ {model_name}_best.pth kaydedildi!")
    # -----------------------
# 7️⃣ KARŞILAŞTIRMALI SONUÇ TABLOSU
# -----------------------
print("\n--- TÜM MODELLER İÇİN KARŞILAŞTIRMALI TABLO ---")
results = []

# ML Modelleri için
for name, model in ml_models.items():
    y_pred = model.predict(X_test)
    results.append({
        'Model': name, 
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'Recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'F1-Score': f1_score(y_test, y_pred, average='macro', zero_division=0)
    })

# DL Modelleri için (ANN, CNN, LSTM)
for model_name, ModelClass in model_classes.items():
    # Modeli diske kaydettiğimiz haliyle geri yükle
    model = ModelClass(input_dim, 64, num_classes, nn.ReLU()).to(device)
    model.load_state_dict(torch.load(f"{model_name}_best.pth"))
    
    # DL değerlendirme
    acc, prec, rec, f1 = evaluate_pytorch_model(model, X_test_tensor, y_test_tensor, model_name)
    results.append({
        'Model': model_name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1
    })

# Tabloyu yazdır
df_summary = pd.DataFrame(results)
print(df_summary.to_string(index=False))
df_summary.to_csv("tum_modeller_ozet.csv", index=False)

joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le, 'label_encoder.pkl')
joblib.dump(df.drop('Label', axis=1).columns.tolist(), 'feature_names.pkl')
print("✅ scaler.pkl, label_encoder.pkl ve feature_names.pkl kaydedildi.")

# Test verilerini de "Live IDS" simülasyonu için kaydedelim
pd.DataFrame(X_test).to_csv("test_ornekleri.csv", index=False)

print("✅ Tüm gerekli dosyalar (scaler, encoder, feature_names, test_ornekleri) kaydedildi!")
