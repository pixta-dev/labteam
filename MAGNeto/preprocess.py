import ast

import pandas as pd
from tqdm import tqdm

from magneto.utils import parse_preprocessing_args


def make_label(tags, important_tags) -> list:
    '''
    input:
        + tags: all available tags of an item.
        + important_tags: tags that marked as important.
    output:
        a binary mask with 0 for unimportant tags and 1 for important ones.
    '''
    return ['1' if tag in important_tags else '0' for tag in tags]


def label_important_tags(
    item_id,
    tags,
    important_tags
) -> dict:
    '''
    input:
        + item_id: the ID of an item.
        + tags: all available tags of an item.
        + important_tags: tags that marked as important.
    output:
        a dictionary which includes all needed information of an item.
    '''
    label = make_label(tags, important_tags)

    return {
        'item_id': item_id,
        'tags': ','.join(tags),
        'important_tags': ','.join(important_tags),
        'label': ','.join(label)
    }


def main():
    opt = parse_preprocessing_args()

    df = pd.read_csv(opt.csv_path)

    assert 'tags' in df.columns
    assert 'important_tags' in df.columns
    assert opt.tags_field_type in ['str', 'list']
    assert opt.important_tags_field_type in ['str', 'list']

    series_of_item_id = df['item_id']
    series_of_tags = df['tags']
    series_of_important_tags = df['important_tags']

    if opt.tags_field_type == 'str':
        series_of_tags = series_of_tags.apply(lambda x: x.split(','))
    elif opt.tags_field_type == 'list':
        series_of_tags = series_of_tags.apply(ast.literal_eval)

    if opt.important_tags_field_type == 'str':
        series_of_important_tags = series_of_important_tags.apply(
            lambda x: x.split(','))
    elif opt.important_tags_field_type == 'list':
        series_of_important_tags = series_of_important_tags.apply(ast.literal_eval)

    rows_dict = dict()
    i = 0

    if opt.use_multiprocessing:
        import multiprocessing as mp

        # Apply a patch for the multiprocessing module
        import multiprocessing.pool as mpp
        from magneto.utils import istarmap
        mpp.Pool.istarmap = istarmap

        if opt.num_workers == -1:
            opt.num_workers = mp.cpu_count()

        inputs = list(zip(
            series_of_item_id,
            series_of_tags,
            series_of_important_tags
        ))
        with mp.Pool(opt.num_workers) as pool:
            for result in tqdm(pool.istarmap(label_important_tags, inputs), total=len(inputs)):
                rows_dict[i] = result
                i += 1
    else:
        for item_id, tags, important_tags \
                in tqdm(list(zip(
                    series_of_item_id,
                    series_of_tags,
                    series_of_important_tags
                ))):

            result = label_important_tags(
                item_id,
                tags,
                important_tags
            )

            rows_dict[i] = result
            i += 1

    new_df = pd.DataFrame.from_dict(rows_dict, 'index')
    new_df.to_csv(opt.save_path, index=False)


if __name__ == '__main__':
    main()
