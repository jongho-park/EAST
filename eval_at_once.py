import json
import os
import os.path as osp
from argparse import ArgumentParser
from glob import glob
from pprint import pprint

from deteval import calc_deteval_metrics


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--gt_dir', default='/nas/ocr/datasets/BoostCamp')
    parser.add_argument('--pred_dir', default='predictions')

    parser.add_argument('--output_dir', default='scores')

    args = parser.parse_args()

    return args


def do_evaluation(pred_ufo, public_gt_ufo, private_gt_ufo):
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

    pred_bboxes = ufo_to_rrc_format(pred_ufo)
    pub_gt_bboxes, pub_transcriptions = ufo_to_rrc_format(public_gt_ufo, get_transcriptions=True)
    pri_gt_bboxes, pri_transcriptions = ufo_to_rrc_format(private_gt_ufo, get_transcriptions=True)

    result_dict = dict()

    pub_pred_bboxes = {x: pred_bboxes[x] for x in pub_gt_bboxes}
    public_result = calc_deteval_metrics(pub_pred_bboxes, pub_gt_bboxes, pub_transcriptions)
    result_dict['public'] = dict(fscore=public_result['total']['hmean'],
                                 recall=public_result['total']['recall'],
                                 precision=public_result['total']['precision'])

    pri_pred_bboxes = {x: pred_bboxes[x] for x in pri_gt_bboxes}
    private_result = calc_deteval_metrics(pri_pred_bboxes, pri_gt_bboxes, pri_transcriptions)
    result_dict['private'] = dict(fscore=private_result['total']['hmean'],
                                  recall=private_result['total']['recall'],
                                  precision=private_result['total']['precision'])

    gt_bboxes = dict(**pub_gt_bboxes, **pri_gt_bboxes)
    transcriptions = dict(**pub_transcriptions, **pri_transcriptions)
    final_result = calc_deteval_metrics(pred_bboxes, gt_bboxes, transcriptions)
    result_dict['final'] = dict(fscore=final_result['total']['hmean'],
                                recall=final_result['total']['recall'],
                                precision=final_result['total']['precision'])

    return result_dict


def main(args):
    with open(osp.join(args.gt_dir, 'public_ufo.json'), 'r') as f:
        public_gt_ufo = json.load(f)
    with open(osp.join(args.gt_dir, 'private_ufo.json'), 'r') as f:
        private_gt_ufo = json.load(f)

    if not osp.exists(args.output_dir):
        os.makedirs(args.output_dir)

    pred_fpaths = glob(osp.join(args.pred_dir, '*.json'))

    for pred_fpath in pred_fpaths:
        pred_name = osp.splitext(osp.basename(pred_fpath))[0]
        print('Evaluation in progress ({})'.format(pred_name))

        with open(pred_fpath, 'r') as f:
            pred_ufo = json.load(f)

        result = do_evaluation(pred_ufo, public_gt_ufo, private_gt_ufo)
        pprint(result)

        with open(osp.join(args.output_dir, '{}.json'.format(pred_name)), 'w') as f:
            json.dump(result, f, indent=4)


if __name__ == '__main__':
    args = parse_args()
    main(args)
