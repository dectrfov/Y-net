import os
import torch
import torch.nn
import torchvision
from PIL import Image
import argparse
from vertical_net import vertical_net


def get_config():

    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--data_path',
                        type=str,
                        help='Testing image path')
    parser.add_argument('-m', '--model', type=str, help='dehazing model')
    config, unparsed = parser.parse_known_args()
    return config, unparsed


def make_test_data(img_path_list, device):
    data_transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()])
    imgs = []
    for img_path in img_path_list:
        x = data_transform(Image.open(str(img_path))).unsqueeze(0)
        x = x.to(device)
        imgs.append(x)
    return imgs


def load_pretrain_network(cfg, device):
    network = vertical_net().to(device)
    weight = torch.load(cfg.model)
    if ('state_dict' in weight):
        network.load_state_dict(weight['state_dict'])
    else:
        network.load_state_dict(weight)
    return network


def main(cfg):
    # -------------------------------------------------------------------
    # basic config

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = cfg.data_path
    # -------------------------------------------------------------------
    # load network
    network = load_pretrain_network(cfg, device)

    imgs = os.listdir(path)
    print('Start eval')
    network.eval()

    for img in imgs:

        test_file_path = os.path.join(path, img)

        test_images = make_test_data([test_file_path], device)
        dehaze_image = network(test_images[0])
        dehaze_image = dehaze_image.to('cpu')
        torchvision.utils.save_image(dehaze_image,
                                     os.path.join(path, 'clear_' + img))


if __name__ == '__main__':

    config_args, unparsed_args = get_config()
    main(config_args)
