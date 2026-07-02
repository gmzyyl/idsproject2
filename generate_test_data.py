import numpy as np

# Kaç örnek üretmek istiyorsun?
num_samples = 5  # 5 örnek

# Özellik sayısı (Label hariç) → senin veri setinde 78
num_features = 78

# Rastgele test verisi
test_X = np.random.rand(num_samples, num_features)  # 0-1 arasında rastgele değerler

print("Test verisi boyutu:", test_X.shape)
print(test_X)