import os
import os.path as osp
import json
from argparse import ArgumentParser
from glob import glob

import torch
import cv2
from torch import cuda
from model import EAST
from tqdm import tqdm

from detect import detect


CHECKPOINT_EXTENSIONS = ['.pth', '.ckpt']


def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', default=os.environ.get('SM_CHANNEL_EVAL'))
    parser.add_argument('--model_dir', default=os.environ.get('SM_CHANNEL_MODEL'))
    parser.add_argument('--output_dir', default=os.environ.get('SM_OUTPUT_DATA_DIR'))

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--input_size', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--of_n_take_ith', type=int, nargs=2, default=[1, 1])

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return parser.parse_args()


def do_inference(model, ckpt_fpath, data_dir, input_size, batch_size, split='public'):
    model.load_state_dict(torch.load(ckpt_fpath, map_location='cpu'))
    model.eval()

    image_fnames, by_sample_bboxes = [], []

    images = []
    for image_fpath in tqdm(glob(osp.join(data_dir, '{}/*'.format(split)))):
        image_fnames.append(osp.basename(image_fpath))

        images.append(cv2.imread(image_fpath)[:, :, ::-1])
        if len(images) == batch_size:
            by_sample_bboxes.extend(detect(model, images, input_size))
            images = []

    if len(images):
        by_sample_bboxes.extend(detect(model, images, input_size))

    ufo_result = dict(images=dict())
    for image_fname, bboxes in zip(image_fnames, by_sample_bboxes):
        words_info = {idx: dict(points=bbox.tolist()) for idx, bbox in enumerate(bboxes)}
        ufo_result['images'][image_fname] = dict(words=words_info)

    return ufo_result


def main(args):
    # Initialize model
    model = EAST(pretrained=False).to(args.device)

    # Get paths to checkpoint files
    ckpt_fnames = sorted([x for x in os.listdir(args.model_dir) if osp.splitext(x)[1] in
                          CHECKPOINT_EXTENSIONS])
    ckpt_fpaths = [osp.join(args.model_dir, x) for x in ckpt_fnames]

    if not osp.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for idx, ckpt_fpath in enumerate(ckpt_fpaths):
        if idx % args.of_n_take_ith[0] != args.of_n_take_ith[1] - 1:
            continue

        ckpt_name = osp.splitext(osp.basename(ckpt_fpath))[0]
        print('Inference in progress ({} | {} / {})'.format(ckpt_name, idx + 1, len(ckpt_fnames)))

        for split in ['public', 'private']:
            print('Split: {}'.format(split))
            ufo_result = do_inference(model, ckpt_fpath, args.data_dir, args.input_size,
                                      args.batch_size, split=split)
            with open(osp.join(args.output_dir, '{}_{}.json'.format(ckpt_name, split)), 'w') as f:
                json.dump(ufo_result, f, indent=4)


if __name__ == '__main__':
    args = parse_args()
    main(args)
