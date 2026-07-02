import matplotlib.pyplot as plt
import pandas as pd

# Verilerini buraya tanımladık
data = {
    'Model': ['LogReg', 'DecTree', 'RandFor', 'SVM', 'KNN', 'AdaBoost', 'GradBoost', 'ANN', 'CNN', 'LSTM'],
    'Accuracy': [0.8717, 0.9716, 0.9690, 0.8783, 0.9528, 0.4894, 0.9627, 0.8931, 0.9061, 0.8965],
    'Precision': [0.8851, 0.9717, 0.9707, 0.9129, 0.9526, 0.4430, 0.9639, 0.9278, 0.9365, 0.9250],
    'Recall': [0.8717, 0.9716, 0.9690, 0.8783, 0.9528, 0.4894, 0.9627, 0.8931, 0.9061, 0.8965],
    'F1-Score': [0.8664, 0.9716, 0.9687, 0.8582, 0.9525, 0.4214, 0.9627, 0.8761, 0.8933, 0.8796]
}

df = pd.DataFrame(data).set_index('Model')

# Çizim
ax = df.plot(kind='bar', figsize=(15, 7), colormap='viridis', edgecolor='black')
plt.title('Modellerin Başarı Metrikleri Karşılaştırması', fontsize=16)
plt.ylabel('Skor', fontsize=12)
plt.ylim(0, 1.1)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)
plt.tight_layout()
plt.show()