import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from common.load_dataset import load_mnist
from deep_cnn import CNN
from common.trainer import Trainer

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)
# ==========================================================
# train_data_num = 5000 
# test_data_num = 1000 
# x_train, t_train = x_train[:train_data_num], t_train[:train_data_num]
# x_test, t_test = x_test[:test_data_num], t_test[:test_data_num]
# print(f"Reduced training data: {train_data_num}, test data: {test_data_num}")
# ==========================================================

network = CNN()
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=5, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr':0.001},
                  evaluate_sample_num_per_epoch=1000)
trainer.train()

# Save parameters
current_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(current_dir, "DeepCNN_params_fast_test.pkl")
network.save_params(save_path)
print("Saved Network Parameters!")