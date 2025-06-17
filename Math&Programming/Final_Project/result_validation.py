import os
import sys
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from common.load_dataset import load_mnist
from deep_cnn import CNN

# 1. Load data
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

# 2. Load trained model
network = CNN()
current_dir = os.path.dirname(os.path.abspath(__file__))
load_path = os.path.join(current_dir, "DeepCNN_params_fast_test.pkl")
network.load_params(load_path)

# 3. Model Accuracy
test_accuracy = network.accuracy(x_test, t_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

# 4. Select sample images
num_samples = 10
x_sample = x_test[:num_samples]
t_sample = t_test[:num_samples]

# 5. Predict
y_pred = network.predict(x_sample)
y_pred_labels = np.argmax(y_pred, axis=1)

# 6. Plot
plt.figure(figsize=(15, 3))
for i in range(num_samples):
    plt.subplot(1, num_samples, i + 1)
    plt.imshow(x_sample[i][0], cmap='gray')
    plt.axis('off')
    plt.title(f"Answer:{t_sample[i]}\nPred:{y_pred_labels[i]}")
plt.tight_layout()

image_path = os.path.join(current_dir, "prediction_results.png")
plt.savefig(image_path)
print(f"Prediction image saved to: {image_path}")
plt.show()