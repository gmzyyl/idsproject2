import joblib
import torch
import pandas as pd
import numpy as np
from model_defs import ANN_Model, CNN_Model, LSTM_Model

# 1. Kaydedilen "Gözlükleri" ve "Anahtarları" yükle
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("label_encoder.pkl")
raw_features = joblib.load("feature_names.pkl")

# 📌 KESİN ÇÖZÜM: feature_names eğer metotsa çalıştır, listeyse doğrudan al
try:
    if callable(raw_features):
        feature_names = list(raw_features())
    else:
        feature_names = list(raw_features)
except Exception:
    # Eğer dosya tamamen bozuksa, yedek plan olarak 78 adet "Feature_X" oluştur
    feature_names = [f"Feature_{i}" for i in range(78)]

def get_predictions(df, model_path, model_type="ML"):
    # 0. Sütun isimlerindeki gizli boşlukları temizle (İşte asıl çözüm bu!)
    df.columns = df.columns.str.strip()
    
    # Orijinal veri setini yüklediğimiz için içinde 'Label' (sonuç) sütunu kalmış olabilir. 
    # Model tahmin yaparken sonucu önceden bilmemeli, o yüzden varsa atıyoruz.
    if 'Label' in df.columns:
        df = df.drop('Label', axis=1)

    # 1. Ham Wireshark paketi mi kontrolü
    mevcut_sutunlar = list(df.columns)
    eksik_sutunlar = [col for col in feature_names if col not in mevcut_sutunlar]
    
    if len(eksik_sutunlar) > 10:
        raise ValueError("Ham Wireshark paketi tespit edildi! Modelin çalışması için bu paketlerin CICFlowMeter (veya benzeri bir araç) kullanılarak 78 adet akış özelliğine (Feature) dönüştürülmesi gerekmektedir.")
    
    # 2. Sütunları eşitle
    df = df[feature_names] 
    
    # 3. Temizlik
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    
    # 4. Ölçekle
    X = scaler.transform(df)
    
    # 5. Tahmin Yap
    if model_type == "ML":
        model = joblib.load(model_path)
        pred_idx = model.predict(X)
    else:
        # DL Modeli yükleme kısmı
        if "ANN" in model_path:
            from model_defs import ANN_Model
            model = ANN_Model(78, 64, 15, torch.nn.ReLU())
        elif "CNN" in model_path:
            from model_defs import CNN_Model
            model = CNN_Model(78, 64, 15, torch.nn.ReLU())
        else:
            from model_defs import LSTM_Model
            model = LSTM_Model(78, 64, 15, torch.nn.ReLU())
            
        model.load_state_dict(torch.load(model_path))
        model.eval()
        with torch.no_grad():
            tensor_x = torch.tensor(X, dtype=torch.float32)
            outputs = model(tensor_x)
            pred_idx = torch.argmax(outputs, dim=1).numpy()
            
    # 6. İsimlere çevir
    return encoder.inverse_transform(pred_idx)