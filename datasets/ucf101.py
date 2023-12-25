import os
import pickle
import re
from .basic import Benchmark, read_split, save_split, read_and_split_data, \
    generate_fewshot_dataset, subsample_classes, split_trainval, read_split_caption
from tools.utils import mkdir_if_missing
from clip import clip


class UCF101(Benchmark):

    dataset_dir = "ucf101"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "UCF-101-midframes")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_UCF101.json")
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
            cname2lab = {}
            filepath = os.path.join(self.dataset_dir, "ucfTrainTestlist/classInd.txt")
            with open(filepath, "r") as f:
                lines = f.readlines()
                for line in lines:
                    label, classname = line.strip().split(" ")
                    label = int(label) - 1  # conver to 0-based index
                    cname2lab[classname] = label

            trainval = self.read_data(cname2lab, "ucfTrainTestlist/trainlist01.txt")
            test = self.read_data(cname2lab, "ucfTrainTestlist/testlist01.txt")
            train, val = split_trainval(trainval)
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

    def read_data(self, cname2lab, text_file):
        text_file = os.path.join(self.dataset_dir, text_file)
        items = []

        with open(text_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")[0]  # trainlist: filename, label
                action, filename = line.split("/")
                label = cname2lab[action]

                elements = re.findall("[A-Z][^A-Z]*", action)
                renamed_action = "_".join(elements)

                filename = filename.replace(".avi", ".jpg")
                impath = os.path.join(self.image_dir, renamed_action, filename)

                item = {'impath': impath,
                        'label': int(label),
                        'classname': renamed_action}
                items.append(item)

        return items
