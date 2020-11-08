import os
import argparse
import multiprocessing as mp
import copy

import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm

from magneto.model import MAGNeto
from magneto.data import TagAndImageDataset
from magneto.augment_helper import val_transform
from magneto.utils import parse_infer_args


def predict(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, opt: argparse.Namespace) -> list:
    '''
    input:
        + model:
        + dataloader:
        + opt: configuration.
    output:
        list of the predictions of all items.
    '''
    all_preds = []
    all_item_ids = []

    with torch.no_grad():
        for batch_val_idx, data in enumerate(tqdm(dataloader)):
            if opt.has_label:
                image_batch, tags_batch, mask_batch, _, item_id_batch = data
            else:
                image_batch, tags_batch, mask_batch, item_id_batch = data
            image_batch = image_batch.to(opt.device)
            tags_batch = tags_batch.to(opt.device)
            mask_batch = mask_batch.to(opt.device)

            preds, _, _, _, _ = model(tags_batch, image_batch, mask_batch)

            preds = preds.detach().cpu().numpy()
            mask_batch = mask_batch.detach().cpu().numpy()
            preds = [tuple(pred[~mask].tolist()) for pred, mask in zip(preds, mask_batch)]

            all_preds.extend(preds)
            all_item_ids.extend(item_id_batch.detach().cpu().tolist())

    return all_preds, all_item_ids


def postprocess_prediction(row, opt: argparse.Namespace):
    '''
    input:
        + row
        + opt: configuration.
    output:
        [important_tags,] post_prediction
    '''
    tags = np.array(row['tags'].split(','))
    tags = tags[:opt.max_len]

    if opt.has_label:
        label = np.array(row['label'].split(','), dtype=np.uint8)
        label = label[:opt.max_len]
        mask = label == 1
        important_tags = tags[mask]

        final_results = sorted(
            zip(tags, row.raw_prediction, mask), key=lambda x: x[1], reverse=True)
    else:
        final_results = sorted(zip(tags, row.raw_prediction),
                               key=lambda x: x[1], reverse=True)

    # Get at least top n important tags
    post_prediction = final_results[:opt.top]
    # Get other accepted important tags based on threshold value
    for final_result in final_results[opt.top:]:
        if final_result[1] > opt.threshold:
            post_prediction.append(final_result)
        else:
            break

    if opt.has_label:
        return important_tags, post_prediction
    else:
        return post_prediction


def postprocess_predictions(df: pd.DataFrame, opt: argparse.Namespace) -> pd.DataFrame:
    '''
    input:
        + df: input pandas dataframe.
        + opt: configuration.
    output:
        postprocessed pandas dataframe.
    '''
    post_predictions = []
    if opt.has_label:
        list_of_important_tags = []

    if opt.use_multiprocessing:
        import multiprocessing as mp

        # Apply a patch for the multiprocessing module
        import multiprocessing.pool as mpp
        from magneto.utils import istarmap
        mpp.Pool.istarmap = istarmap

        all_rows = [row for idx, row in df.iterrows()]

        inputs = list(zip(
            all_rows,
            [copy.deepcopy(opt) for _ in range(len(df))]
        ))

        with mp.Pool(opt.num_workers) as pool:
            for result in tqdm(pool.istarmap(postprocess_prediction, inputs), total=len(inputs)):
                if opt.has_label:
                    important_tags, post_prediction = result
                    list_of_important_tags.append(important_tags)
                else:
                    post_prediction = result

                post_predictions.append(post_prediction)

    else:
        for idx, row in tqdm(list(df.iterrows())):
            if opt.has_label:
                important_tags, post_prediction = postprocess_prediction(
                    row, opt)
                list_of_important_tags.append(important_tags)
            else:
                post_prediction = postprocess_prediction(
                    row, opt)

            post_predictions.append(post_prediction)

    list_of_pred_tags = []
    list_of_probs = []

    for post_prediction in post_predictions:
        post_prediction = list(zip(*post_prediction))
        if len(post_prediction) >= 2:
            # TODO we will take care of masks later.
            pred_tags, probs = post_prediction[0], post_prediction[1]

            list_of_pred_tags.append('\n'.join(pred_tags))
            probs = np.round(probs, decimals=3)
            probs = np.array(probs, dtype=str)
            list_of_probs.append('\n'.join(probs))
        else:
            list_of_pred_tags.append('')
            list_of_probs.append('')

    df['pred_tags'] = list_of_pred_tags
    df['probs'] = list_of_probs

    if opt.has_label:
        list_of_important_tags = list(
            map(lambda x: '\n'.join(x), list_of_important_tags))

        df['important_tags'] = list_of_important_tags

    return df


def main():
    opt = parse_infer_args()

    states = torch.load(
        opt.model_path, map_location=lambda storage, loc: storage)

    # Load model's configuration
    model_config = states['config']
    opt.max_len = model_config['max_len']
    opt.d_model = model_config['d_model']
    opt.t_blocks = model_config['t_blocks']
    opt.t_heads = model_config['t_heads']
    opt.t_dim_feedforward = model_config['t_dim_feedforward']
    opt.i_blocks = model_config['i_blocks']
    opt.i_heads = model_config['i_heads']
    opt.i_dim_feedforward = model_config['i_dim_feedforward']
    opt.img_backbone = model_config['img_backbone']
    opt.g_dim_feedforward = model_config['g_dim_feedforward']

    test_dataset = TagAndImageDataset(
        csv_path=opt.csv_path,
        vocab_path=opt.vocab_path,
        img_dir=opt.img_dir,
        max_len=opt.max_len,
        has_label=opt.has_label,
        return_item_id=True,
        img_preprocess_fn=val_transform
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        pin_memory=True if not opt.no_cuda else False
    )
    model = MAGNeto(
        d_model=opt.d_model,
        vocab_size=test_dataset.vocab_size,
        t_blocks=opt.t_blocks,
        t_heads=opt.t_heads,
        t_dim_feedforward=opt.t_dim_feedforward,
        i_blocks=opt.i_blocks,
        i_heads=opt.i_heads,
        i_dim_feedforward=opt.i_dim_feedforward,
        img_backbone=opt.img_backbone,
        g_dim_feedforward=opt.g_dim_feedforward,
        dropout=0
    )
    model.load_state_dict(states['model'])
    model.to(opt.device)
    model.eval()

    all_preds, all_item_ids = predict(model, test_dataloader, opt)
    raw_prediction_df = pd.DataFrame({
        'item_id': all_item_ids,
        'raw_prediction': all_preds
    }).drop_duplicates().set_index('item_id')

    base_df = pd.read_csv(opt.csv_path, index_col='item_id')

    # Log all error item ids
    error_item_ids = np.setdiff1d(base_df.index.unique(), raw_prediction_df.index.unique(), assume_unique=True).astype(str)
    if len(error_item_ids) > 0:
        print('Error item ids:', ', '.join(error_item_ids))
        with open('error_item_ids.txt', 'w') as f:
            f.write('\n'.join(error_item_ids))

    final_df = raw_prediction_df.join(base_df).reset_index()

    final_df = postprocess_predictions(final_df, opt)

    if opt.has_label:
        final_df.rename(columns={'important_tags': 'ground_truth'}, inplace=True)
        final_df[['item_id', 'tags', 'pred_tags', 'probs', 'ground_truth']].to_csv(
            'prediction.csv', index=False)
    else:
        final_df[['item_id', 'tags', 'pred_tags', 'probs']].to_csv(
            'prediction.csv', index=False)


if __name__ == '__main__':
    main()
