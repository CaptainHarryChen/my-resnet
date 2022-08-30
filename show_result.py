import os
import numpy as np
import torch

import matplotlib.pyplot as plt


def show_result(record_name):
    record_dict = torch.load(os.path.join("records", record_name))
    plt.figure()
    plt.title("Losses")
    plt.ylabel("loss")
    plt.xlabel("train iteration")
    plt.plot(
        np.arange(1, record_dict["train_iterations"]+1), record_dict["train_losses"], label="train")
    plt.plot(np.arange(record_dict["save_per_iter"], record_dict["train_iterations"] +
             1, record_dict["save_per_iter"]), record_dict["val_losses"], label="validation")
    plt.legend(loc="upper right")

    plt.figure()
    plt.title("Validation Accuracy")
    plt.plot(np.arange(record_dict["save_per_iter"], record_dict["train_iterations"] +
             1, record_dict["save_per_iter"]), record_dict["val_accuracy"])

    plt.show()


if __name__ == "__main__":
    show_result("cifar_record_1.pth")
