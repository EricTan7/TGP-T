import os
import pickle
import random
from scipy.io import loadmat
from collections import defaultdict
from .basic import Benchmark, read_split, save_split, read_and_split_data, \
    generate_fewshot_dataset, subsample_classes, read_split_caption
from tools.utils import load_json, mkdir_if_missing
from clip import clip


class OxfordFlowers(Benchmark):

    dataset_dir = "oxford_flowers"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "jpg")
        self.label_file = os.path.join(self.dataset_dir, "imagelabels.mat")
        self.lab2cname_file = os.path.join(self.dataset_dir, "cat_to_name.json")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_OxfordFlowers.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot2")

        caption = dict()
        tokenized_caption = dict()
        caption_path = os.path.join(self.dataset_dir, "captions.txt")
        with open(caption_path, 'r') as f:
            for line in f.readlines():
                line = line.strip('\n').split('\t')
                caption[line[0]] = line[1]
                tokenized_caption[line[0]] = clip.tokenize(line[1])[0]

        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            train, val, test = read_split_caption(self.split_path, self.image_dir, caption, tokenized_caption)
        else:
            train, val, test = self.read_data()
            save_split(train, val, test, self.split_path, self.image_dir)

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = generate_fewshot_dataset(train, num_shots=num_shots)['data']
                val = generate_fewshot_dataset(val, num_shots=min(num_shots, 4))['data']
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        super().__init__(train=train, val=val, test=test)

    def read_data(self):
        tracker = defaultdict(list)
        label_file = loadmat(self.label_file)["labels"][0]
        for i, label in enumerate(label_file):
            imname = f"image_{str(i + 1).zfill(5)}.jpg"
            impath = os.path.join(self.image_dir, imname)
            label = int(label)
            tracker[label].append(impath)

        print("Splitting data into 50% train, 20% val, and 30% test")

        def _collate(ims, y, c):
            items = []
            for im in ims:
                # convert to 0-based label
                item = {'impath': im,
                        'label': y-1,
                        'classname': c}
                items.append(item)
            return items

        lab2cname = load_json(self.lab2cname_file)
        train, val, test = [], [], []
        for label, impaths in tracker.items():
            random.shuffle(impaths)
            n_total = len(impaths)
            n_train = round(n_total * 0.5)
            n_val = round(n_total * 0.2)
            n_test = n_total - n_train - n_val
            assert n_train > 0 and n_val > 0 and n_test > 0
            cname = lab2cname[str(label)]
            train.extend(_collate(impaths[:n_train], label, cname))
            val.extend(_collate(impaths[n_train : n_train + n_val], label, cname))
            test.extend(_collate(impaths[n_train + n_val :], label, cname))

        return train, val, test
