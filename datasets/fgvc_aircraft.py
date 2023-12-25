import os
import pickle
from .basic import Benchmark, read_split, save_split, read_and_split_data, \
    generate_fewshot_dataset, subsample_classes, read_split_caption
from tools.utils import mkdir_if_missing
from clip import clip


class FGVCAircraft(Benchmark):

    dataset_dir = "fgvc_aircraft"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot2")
        mkdir_if_missing(self.split_fewshot_dir)

        classnames = []
        with open(os.path.join(self.dataset_dir, "variants.txt"), "r") as f:
            lines = f.readlines()
            for line in lines:
                classnames.append(line.strip())
        cname2lab = {c: i for i, c in enumerate(classnames)}

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
                test = self.read_data_caption(cname2lab, "images_variant_test.txt")

            else:
                caption = dict()
                tokenized_caption = dict()
                caption_path = os.path.join(self.dataset_dir, "captions.txt")
                with open(caption_path, 'r') as f:
                    for line in f.readlines():
                        line = line.strip('\n').split('\t')
                        caption[line[0]] = line[1]
                        tokenized_caption[line[0]] = clip.tokenize(line[1])[0]
                train = self.read_data_caption(cname2lab, "images_variant_train.txt", caption, tokenized_caption)
                val = self.read_data_caption(cname2lab, "images_variant_val.txt")
                test = self.read_data_caption(cname2lab, "images_variant_test.txt")
                train = generate_fewshot_dataset(train, num_shots=num_shots)['data']
                val = generate_fewshot_dataset(val, num_shots=min(num_shots, 4))['data']
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        super().__init__(train=train, val=val, test=test)

    def read_data(self, cname2lab, split_file):
        filepath = os.path.join(self.dataset_dir, split_file)
        items = []

        with open(filepath, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                imname = line[0] + ".jpg"
                classname = " ".join(line[1:])
                impath = os.path.join(self.image_dir, imname)
                label = cname2lab[classname]
                item = {'impath': impath,
                        'label': int(label),
                        'classname': classname}
                items.append(item)

        return items

    def read_data_caption(self, cname2lab, split_file, caption=None, tokenized_caption=None):
        filepath = os.path.join(self.dataset_dir, split_file)
        items = []

        with open(filepath, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                imname = line[0] + ".jpg"
                classname = " ".join(line[1:])
                impath = os.path.join(self.image_dir, imname)
                label = cname2lab[classname]
                item = {'impath': impath,
                        'label': int(label),
                        'classname': classname,
                        'caption': caption[imname] if caption is not None else None,
                        'tokenized_caption': tokenized_caption[imname] if tokenized_caption is not None else None
                        }
                items.append(item)

        return items
