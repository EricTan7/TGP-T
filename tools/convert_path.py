import json
import pickle
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Path Converting")
    parser.add_argument(
        "--input-file", default="", help="path to config file", type=str
    )
    parser.add_argument(
        "--org-path", default="/mnt/nas/TrueNas1/tanhao/recognition", help="path to be converted", type=str
    )
    parser.add_argument(
        "--cvt-path",
        help='convert path', type=str, default="",
    )
    args = parser.parse_args()

    with open(args.input_file, 'rb') as f:
        data = pickle.load(f)
        train, val = data["train"], data["val"]
        for i in range(len(train)):
            path_org = train[i]['impath']
            path_cvt = path_org.replace(args.org_path, args.cvt_path)
            train[i]['impath'] = path_cvt
        for i in range(len(val)):
            path_org = val[i]['impath']
            path_cvt = path_org.replace(args.org_path, args.cvt_path)
            val[i]['impath'] = path_cvt
        data = {'train': train, 'val': val}
    
    with open(args.input_file, "wb") as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

    print("Converted file has been saved to ==> {}".format(args.input_file))
