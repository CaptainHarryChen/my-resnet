import os
import json
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision.io import read_image

class MiniImagenet(Dataset):
    def __init__(self, root, split: str="train", transform=None, augmentation=None):
        processed_path = os.path.join(root, "processed")
        if not os.path.exists(processed_path):
            process(root)
        
        self.img_labels = pd.read_csv(os.path.join(processed_path, split+".csv"))
        self.img_dir = os.path.join(root, "images")
        self.label_dict = json.load(open(os.path.join(processed_path, "class_index.json"), "r"))
        self.transform = transform
        self.augmentation = augmentation
        
        self.class_num = len(self.label_dict)


    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        image = image / 256.0
        label = self.img_labels.iloc[idx, 1]
        label = self.label_dict[label][0]
        if self.augmentation:
            image = self.augmentation(image)
        if self.transform:
            image = self.transform(image)
        return image, label


def process(root: str, val_rate = 0.1, test_rate = 0.1):
    raw_train_data = pd.read_csv(os.path.join(root, "train.csv"))
    raw_test_data = pd.read_csv(os.path.join(root, "test.csv"))
    raw_val_data = pd.read_csv(os.path.join(root, "val.csv"))

    data = pd.concat([raw_train_data, raw_test_data, raw_val_data], axis=0)
    labels = data["label"].unique()
    # print(labels)
    raw_label_dict = json.load(open(os.path.join(root, "imagenet_class_index.json"), "r"))
    raw_label_dict = {v[0]:v[1] for k, v in raw_label_dict.items()}

    label_dict = {lab:(i,raw_label_dict[lab]) for i,lab in enumerate(labels)}
    # print(label_dict)

    train_data = []
    val_data = []
    test_data = []
    for label in labels:
        class_data = data[data["label"] == label]
        # print(f"Class '{label_dict[label][1]}' has {len(class_data)} images.")
        num_val = int(val_rate * len(class_data))
        num_test = int(test_rate * len(class_data))

        class_data = class_data.sample(frac=1.0)
        val_data.append(class_data[:num_val])
        test_data.append(class_data[num_val:num_val+num_test])
        train_data.append(class_data[num_val+num_test:])
        # show_img(root, class_data.iloc[0])
    train_data = pd.concat(train_data, axis=0)
    val_data = pd.concat(val_data, axis=0)
    test_data = pd.concat(test_data, axis=0)
    print(f"train data size: {len(train_data)}")
    print(f"val data size: {len(val_data)}")
    print(f"test data size: {len(test_data)}")

    out_path = os.path.join(root, "processed")
    os.makedirs(out_path)
    train_data.to_csv(os.path.join(out_path, "train.csv"), index=0)
    val_data.to_csv(os.path.join(out_path, "val.csv"), index=0)
    test_data.to_csv(os.path.join(out_path, "test.csv"), index=0)
    with open(os.path.join(out_path, "class_index.json"), 'w') as json_file:
        json_file.write(json.dumps(label_dict, indent=4))


def show_img(root, data):
    img_path = os.path.join(root, "images", data["filename"])
    img = read_image(img_path)
    npimg = img.numpy()
    import matplotlib.pyplot as plt
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()
