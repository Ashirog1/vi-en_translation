import os, io
from torchtext.data.datasets_utils import (
    _RawTextIterableDataset,
)

NUM_LINES_EnVi = {
    'train': 20000,
    # 'train': 10000,
    'dev': 1007,
    'tst': 1220,
}

DATASET_NAME_EnVi = "VLSP2021"

def _read_text_iterator(path, prefix=''):
    with io.open(path, encoding="utf8") as f:
        for row in f:
            if prefix != '':
                yield prefix + ' : ' + row
            else:
                yield row

def VLSP2021(root, split, language_pair=('en', 'vi'), prefix=''):
    """
    Args:
        root: Directory where the datasets are saved. Default: ".data"
        split: split or splits to be returned. Can be a string or tuple of strings. Default: ('train', 'valid', 'test')
        language_pair: tuple or list containing src and tgt language. Available options are ('de','en') and ('en', 'de')
    """

    assert (len(language_pair) == 2), 'language_pair must contain only 2 elements: src and tgt language respectively'

    src_data_iter = _read_text_iterator(os.path.join(root, '{}.{}'.format(split,language_pair[0])), prefix=prefix)
    trg_data_iter = _read_text_iterator(os.path.join(root, '{}.{}'.format(split,language_pair[1])), prefix=prefix)

    return _RawTextIterableDataset(DATASET_NAME_EnVi, NUM_LINES_EnVi[split], zip(src_data_iter, trg_data_iter))

def VLSP2021_DEP(root, split, language_pair=('en', 'vi'), prefix=''):
    """
    Args:
        root: Directory where the datasets are saved. Default: ".data"
        split: split or splits to be returned. Can be a string or tuple of strings. Default: ('train', 'valid', 'test')
        language_pair: tuple or list containing src and tgt language. Available options are ('de','en') and ('en', 'de')
    """

    assert (len(language_pair) == 2), 'language_pair must contain only 2 elements: src and tgt language respectively'

    src_data_iter = _read_text_iterator(os.path.join(root, '{}.{}_dep'.format(split,language_pair[0])), prefix=prefix)
    trg_data_iter = _read_text_iterator(os.path.join(root, '{}.{}_dep'.format(split,language_pair[1])), prefix=prefix)

    return _RawTextIterableDataset(DATASET_NAME_EnVi, NUM_LINES_EnVi[split], zip(src_data_iter, trg_data_iter))

def VLSP2021_HEAD(root, split, language_pair=('en', 'vi'), prefix=''):
    """
    Args:
        root: Directory where the datasets are saved. Default: ".data"
        split: split or splits to be returned. Can be a string or tuple of strings. Default: ('train', 'valid', 'test')
        language_pair: tuple or list containing src and tgt language. Available options are ('de','en') and ('en', 'de')
    """

    assert (len(language_pair) == 2), 'language_pair must contain only 2 elements: src and tgt language respectively'

    src_data_iter = _read_text_iterator(os.path.join(root, '{}.{}_head'.format(split,language_pair[0])), prefix=prefix)
    trg_data_iter = _read_text_iterator(os.path.join(root, '{}.{}_head'.format(split,language_pair[1])), prefix=prefix)

    return _RawTextIterableDataset(DATASET_NAME_EnVi, NUM_LINES_EnVi[split], zip(src_data_iter, trg_data_iter))