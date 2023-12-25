import os
import pickle
from scipy.io import loadmat
from .basic import Benchmark, read_split, save_split, read_and_split_data, \
    generate_fewshot_dataset, subsample_classes, split_trainval, read_split_caption
from tools.utils import mkdir_if_missing
from clip import clip


class StanfordCars(Benchmark):

    dataset_dir = "stanford_cars"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_StanfordCars.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot2")

        caption = dict()
        tokenized_caption = dict()
        caption_path = os.path.join(self.dataset_dir, "captions.txt")
        with open(caption_path, 'r') as f:
            for line in f.readlines():
                line = line.strip('\n').split('\t')
                # cars_train/00467.jpg
                cname = os.path.join("cars_train", line[0])
                caption[cname] = line[1]
                tokenized_caption[cname] = clip.tokenize(line[1])[0]

        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            train, val, test = read_split_caption(self.split_path, self.dataset_dir, caption, tokenized_caption)
        else:
            trainval_file = os.path.join(self.dataset_dir, "devkit", "cars_train_annos.mat")
            test_file = os.path.join(self.dataset_dir, "cars_test_annos_withlabels.mat")
            meta_file = os.path.join(self.dataset_dir, "devkit", "cars_meta.mat")
            trainval = self.read_data("cars_train", trainval_file, meta_file)
            test = self.read_data("cars_test", test_file, meta_file)
            train, val = split_trainval(trainval)
            save_split(train, val, test, self.split_path, self.dataset_dir)

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

    def read_data(self, image_dir, anno_file, meta_file):
        anno_file = loadmat(anno_file)["annotations"][0]
        meta_file = loadmat(meta_file)["class_names"][0]
        items = []

        for i in range(len(anno_file)):
            imname = anno_file[i]["fname"][0]
            impath = os.path.join(self.dataset_dir, image_dir, imname)
            label = anno_file[i]["class"][0, 0]
            label = int(label) - 1  # convert to 0-based index
            classname = meta_file[label][0]
            names = classname.split(" ")
            year = names.pop(-1)
            names.insert(0, year)
            classname = " ".join(names)
            item = {'impath': impath,
                    'label': int(label),
                    'classname': classname}
            items.append(item)

        return items
