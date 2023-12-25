import os
import pickle
from .basic import Benchmark, read_split, save_split, read_and_split_data, \
    generate_fewshot_dataset, subsample_classes, read_split_caption
from tools.utils import mkdir_if_missing
from clip import clip


class OxfordPets(Benchmark):

    dataset_dir = "oxford_pets"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.anno_dir = os.path.join(self.dataset_dir, "annotations")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_OxfordPets.json")
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

        train, val, test = read_split_caption(self.split_path, self.image_dir, caption, tokenized_caption)

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

    def read_data(self, split_file):
        filepath = os.path.join(self.anno_dir, split_file)
        items = []

        with open(filepath, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                imname, label, species, _ = line.split(" ")
                breed = imname.split("_")[:-1]
                breed = "_".join(breed)
                breed = breed.lower()
                imname += ".jpg"
                impath = os.path.join(self.image_dir, imname)
                label = int(label) - 1  # convert to 0-based index
                item = {'impath': impath,
                        'label': label,
                        'classname': breed}
                items.append(item)

        return items