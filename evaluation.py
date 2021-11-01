import json
import os.path as osp
from argparse import ArgumentParser
from glob import glob
from pprint import pprint

from deteval import calc_deteval_metrics


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--gt_path', default='/nas/ocr/datasets/BoostCamp/public_ufo.json')
    parser.add_argument('--pred_path', default='./predictions/model_epoch_100.json')

    args = parser.parse_args()

    return args


def evaluation(gt_path, pred_path):
    """
    Args:
        gt_path (string) : Ground truth file path
        pred_path (string) : Prediction file path (output of inference.py)
    """
    def ufo_to_rrc_format(ufo, get_transcriptions=False):
        bboxes_dict, transcriptions_dict = dict(), dict()
        for image_fname in ufo['images']:
            bboxes, transcriptions = [], []

            for word_info in ufo['images'][image_fname]['words'].values():
                bboxes.append(word_info['points'])
                if get_transcriptions:
                    if word_info.get('illegibility', False):
                        transcriptions.append('###')
                    else:
                        transcriptions.append(word_info['transcription'])

            bboxes_dict[image_fname] = bboxes
            if get_transcriptions:
                transcriptions_dict[image_fname] = transcriptions

        if get_transcriptions:
            return bboxes_dict, transcriptions_dict
        else:
            return bboxes_dict

    with open(gt_path, 'r') as f:
        gt_ufo = json.load(f)
    gt_bboxes, transcriptions = ufo_to_rrc_format(gt_ufo, get_transcriptions=True)

    with open(pred_path, 'r') as f:
        pred_ufo = json.load(f)
    pred_bboxes = ufo_to_rrc_format(pred_ufo)
    pred_bboxes = {x: pred_bboxes[x] for x in gt_bboxes}

    print(len(gt_bboxes), len(pred_bboxes))

    eval_result = calc_deteval_metrics(pred_bboxes, gt_bboxes, transcriptions)

    result_dict = dict(
        f1=dict(
            value=eval_result['total']['hmean'],
            rank=True,
            decs=True,
        ),
        recall=dict(
            value=eval_result['total']['recall'],
            rank=False,
            decs=True,
        ),
        precision=dict(
            value=eval_result['total']['precision'],
            rank=False,
            decs=True,
        ),
    )

    return json.dumps(result_dict)


def main(args):
    result = evaluation(args.gt_path, args.pred_path)
    pprint(result)


if __name__ == '__main__':
    args = parse_args()
    main(args)
