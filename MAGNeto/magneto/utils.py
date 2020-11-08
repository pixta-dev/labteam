import os
import argparse
import multiprocessing as mp
import multiprocessing.pool as mpp

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from magneto.loss import BCEDiceLoss
from magneto.metrics import PrecisionRecallFk


def istarmap(self, func, iterable, chunksize=1):
    ''' starmap-version of imap
    '''
    if self._state != mpp.RUN:
        raise ValueError("Pool not running")

    if chunksize < 1:
        raise ValueError(
            "Chunksize must be 1+, not {0:n}".format(
                chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    result = mpp.IMapIterator(self._cache)
    self._taskqueue.put(
        (
            self._guarded_task_generation(result._job,
                                          mpp.starmapstar,
                                          task_batches),
            result._set_length
        ))
    return (item for chunk in result for item in chunk)


def moving_avg(avg, update, alpha):
    return (alpha * avg) + ((1 - alpha) * update)


def parse_train_args() -> argparse.Namespace:
    '''
    output:
        parsed arguments.
    '''
    parser = argparse.ArgumentParser(description='MAGNeto training process.')
    parser.add_argument(
        '--train-csv-path',
        type=str,
        help='[/path/to/train_data.csv]',
        required=True
    )
    parser.add_argument(
        '--val-csv-path',
        type=str,
        help='[/path/to/val_data.csv]',
        required=True
    )
    parser.add_argument(
        '--vocab-path',
        type=str,
        help='[/path/to/vocab.csv]',
        required=True
    )
    parser.add_argument(
        '--img-dir',
        type=str,
        help='[/path/to/img_dir]',
        required=False
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        help='[/path/to/save_dir]',
        required=True
    )
    parser.add_argument(
        '--checkpoint-path',
        type=str,
        help='[/path/to/checkpoint.pth]'
    )
    parser.add_argument(
        '--load-weights-only',
        action='store_true',
        help='Only does load model\'s weights from checkpoint.'
    )
    parser.add_argument(
        '--exclude-top',
        action='store_true',
        help='Whether excluding top layers or not when loading checkpoint.'
    )
    parser.add_argument(
        '--start-from-epoch',
        type=int,
        help='(default: "0".)',
        default=0
    )
    parser.add_argument(
        '--max-len',
        type=int,
        help='The maximum length for each set of tags (default: 100).',
        default=100
    )
    parser.add_argument(
        '--t-heads',
        type=int,
        help='The number of heads of each Multi-Head Attention layer of the tag branch (default: 8).',
        default=8
    )
    parser.add_argument(
        '--t-blocks',
        type=int,
        help='The number of encoder layers, or blocks, for tag branch (default: 6).',
        default=6
    )
    parser.add_argument(
        '--t-dim-feedforward',
        type=int,
        help='The dimension of the feedforward network model in the TransformerEncoderLayer class of the tag branch, (default: 2048).',
        default=2048
    )
    parser.add_argument(
        '--i-heads',
        type=int,
        help='The number of heads of each Multi-Head Attention layer of the image branch (default: 8).',
        default=8
    )
    parser.add_argument(
        '--i-blocks',
        type=int,
        help='The number of encoder layers, or blocks, for image branch (default: 6).',
        default=2
    )
    parser.add_argument(
        '--i-dim-feedforward',
        type=int,
        help='The dimension of the feedforward network model in the TransformerEncoderLayer of the image branch class, (default: 2048).',
        default=2048
    )
    parser.add_argument(
        '--d-model',
        type=int,
        help='The dimentionality of a context vector, must be divisible by the number of heads, (default: 512).',
        default=512
    )
    parser.add_argument(
        '--img-backbone',
        type=str,
        help='resnet18 or resnet50, (default: resnet50).',
        default='resnet50'
    )
    parser.add_argument(
        '--g-dim-feedforward',
        type=int,
        help='The dimension of the feedforward network model in the GatingLayer class, (default: 2048).',
        default=2048
    )
    parser.add_argument(
        '--dropout',
        type=float,
        help='Dropout value of tag encoder layers (default: 0.1).',
        default=0.1
    )
    parser.add_argument(
        '--tagaug-add-max-ratio',
        type=float,
        help='The maximum ratio between the number of adding tags and non-important ones, (default: 0.3).',
        default=0.3
    )
    parser.add_argument(
        '--tagaug-drop-max-ratio',
        type=float,
        help='The maximum ratio between the number of dropping tags and non-important ones, (default: 0.3).',
        default=0.3
    )
    parser.add_argument(
        '--train-batch-size',
        type=int,
        help='The batch size used in the training process (default: 64).',
        default=64
    )
    parser.add_argument(
        '--val-batch-size',
        type=int,
        help='The batch size used in the validation process (default: 128).',
        default=128
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        help='The number of workers used for data loaders, \
            -1 means using all available processors, \
            rules of thumb: num_workers ~ num_gpu * 4, \
            (default: 4).',
        default=4
    )
    parser.add_argument(
        '--epochs',
        type=int,
        required=True
    )
    parser.add_argument(
        '--lr',
        type=float,
        help='(default: "3e-2".)',
        default=3e-2
    )
    parser.add_argument(
        '--threshold',
        type=float,
        help='(default: "0.5".)',
        default=0.5
    )
    parser.add_argument(
        '--no-cuda',
        action='store_true'
    )
    parser.add_argument(
        '--gpu-id',
        type=int,
        help='The ID of selected GPU, --no-cuda must be disabled, (default: 0).',
        default=0
    )
    parser.add_argument(
        '--use-steplr-scheduler',
        action='store_true',
        help='Whether or not to use StepLR scheduler. \
            all other schedulers should should be disabled.'
    )
    parser.add_argument(
        '--sl-gamma',
        type=float,
        help='StepLR-scheduler\'s multiplicative factor of learning rate decay (default: 0.9).',
        default=0.9
    )
    parser.add_argument(
        '--use-rop-scheduler',
        action='store_true',
        help='Whether or not to use ReduceLROnPlateau scheduler. \
            all other schedulers should should be disabled.'
    )
    parser.add_argument(
        '--rop-factor',
        type=float,
        help='ReduceLROnPlateau scheduler\'s factor parameter (default: 0.3).',
        default=0.3
    )
    parser.add_argument(
        '--rop-patience',
        type=int,
        help='ReduceLROnPlateau scheduler\'s patience parameter (default: 3).',
        default=3
    )
    parser.add_argument(
        '--log-graph',
        action='store_true',
        help='Write down model graph.'
    )
    parser.add_argument(
        '--save-latest',
        action='store_true',
        help='Save the latest checkpoint.'
    )
    parser.add_argument(
        '--save-best-f1',
        action='store_true',
        help='Save the checkpoint based on val F1.'
    )
    parser.add_argument(
        '--save-best-loss',
        action='store_true',
        help='Save the checkpoint based on val loss.'
    )
    parser.add_argument(
        '--save-all-epochs',
        action='store_true',
        help='Save a checkpoint for each epoch.'
    )
    parser.add_argument(
        '--log-weight-hist',
        action='store_true',
        help='Log the histogram of image and tag weights during the validation process.'
    )

    opt = parser.parse_args()

    # Check configuration
    assert not (opt.use_steplr_scheduler and opt.use_rop_scheduler), \
        'Cannot use multiple schedulers at the same time!'

    if opt.num_workers == -1:
        opt.num_workers = mp.cpu_count()

    opt.device = 'cuda:{0}'.format(opt.gpu_id) if not opt.no_cuda else 'cpu'
    if not opt.no_cuda:
        assert torch.cuda.is_available()

    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)

    assert os.path.isfile(opt.train_csv_path)
    assert os.path.isfile(opt.val_csv_path)
    assert os.path.exists(opt.img_dir)

    return opt


def parse_infer_args() -> argparse.Namespace:
    '''
    output:
        parsed arguments.
    '''
    parser = argparse.ArgumentParser(description='Inference module.')
    parser.add_argument(
        '--csv-path',
        type=str,
        help='[/path/to/data.csv]',
        required=True
    )
    parser.add_argument(
        '--img-dir',
        type=str,
        help='[/path/to/img_dir]',
        required=True
    )
    parser.add_argument(
        '--vocab-path',
        type=str,
        help='[/path/to/vocab.csv]',
        required=True
    )
    parser.add_argument(
        '--model-path',
        type=str,
        help='[/path/to/model.pth]',
        required=True
    )
    parser.add_argument(
        '--has-label',
        action='store_true'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        help='The batch size used in the inference process (default: 64).',
        default=64
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        help='The number of workers used for data loaders, \
            -1 means using all available processors, \
            rules of thumb: num_workers ~ num_gpu * 4, \
            (default: 4).',
        default=4
    )
    parser.add_argument(
        '--threshold',
        type=float,
        help='(default: "0.5".)',
        default=0.5
    )
    parser.add_argument(
        '--top',
        type=int,
        help='The minimum number of selected important tags for each item (default: 5).',
        default=5
    )
    parser.add_argument(
        '--no-cuda',
        action='store_true'
    )
    parser.add_argument(
        '--gpu-id',
        type=int,
        help='The ID of selected GPU, --no-cuda must be disabled, (default: 0).',
        default=0
    )
    parser.add_argument(
        '-m',
        '--use-multiprocessing',
        action='store_true',
        help='Activate multiprocessing.'
    )

    opt = parser.parse_args()

    # Check configuration
    if opt.num_workers == -1:
        opt.num_workers = mp.cpu_count()

    opt.device = 'cuda:{0}'.format(opt.gpu_id) if not opt.no_cuda else 'cpu'
    if not opt.no_cuda:
        assert torch.cuda.is_available()

    return opt


def parse_preprocessing_args() -> argparse.Namespace:
    '''
    output:
        parsed arguments.
    '''
    parser = argparse.ArgumentParser(
        description='Generating labels for "tags" and "important_tags" pairs.')
    parser.add_argument(
        '-c',
        '--csv-path',
        type=str,
        help='/path/to/raw_data.csv',
        required=True
    )
    parser.add_argument(
        '-s',
        '--save-path',
        type=str,
        help='/path/to/result.csv',
        default='./result.csv'
    )
    parser.add_argument(
        '-tt',
        '--tags-field-type',
        type=str,
        help='str or list (default: str).',
        default='str'
    )
    parser.add_argument(
        '-it',
        '--important-tags-field-type',
        type=str,
        help='str or list (default: str).',
        default='str'
    )
    parser.add_argument(
        '-m',
        '--use-multiprocessing',
        action='store_true',
        help='Activate multiprocessing.'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        help='The number of workers used for data loaders, -1 means using all available processors, (default: -1).',
        default=-1
    )

    return parser.parse_args()


def parse_pseudo_label_args() -> argparse.Namespace:
    '''
    output:
        parsed arguments.
    '''
    parser = argparse.ArgumentParser(description='Pseudo labeling module.')
    parser.add_argument(
        '--csv-path',
        type=str,
        help='[/path/to/data.csv]',
        required=True
    )
    parser.add_argument(
        '--img-dir',
        type=str,
        help='[/path/to/img_dir]',
        required=True
    )
    parser.add_argument(
        '--model-path',
        type=str,
        help='[/path/to/model.pth]',
        required=True
    )
    parser.add_argument(
        '--save-path',
        type=str,
        help='[/path/to/result.csv]',
        required=True
    )
    parser.add_argument(
        '--item-id-field',
        type=str,
        help='(default: "item_id".)',
        default='item_id'
    )
    parser.add_argument(
        '--tags-field',
        type=str,
        help='(default: "tags".)',
        default='tags'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        help='The batch size used in the inference process (default: 64).',
        default=64
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        help='The number of workers used for data loaders, \
            -1 means using all available processors, \
            rules of thumb: num_workers ~ num_gpu * 4, \
            (default: 4).',
        default=4
    )
    parser.add_argument(
        '--threshold',
        type=float,
        help='(default: "0.5".)',
        default=0.5
    )
    parser.add_argument(
        '--pos-threshold',
        type=float,
        help='The threshold used to classify an item into positive or non-positive class, \
        an item with a score higher than the threshold will be considered a positive sample (default: 0.95).',
        default=0.95
    )
    parser.add_argument(
        '--neg-threshold',
        type=float,
        help='The threshold used to classify an item into negative or non-negative class, \
        an item with a score lower than the threshold will be considered a negative sample (default: 0.05).',
        default=0.05
    )
    parser.add_argument(
        '--max-ratio',
        type=float,
        help='The maximum value for the ratio of the number of the confident tags to the number of all tags (default: 0.05).',
        default=0.05
    )
    parser.add_argument(
        '--min-positive',
        type=int,
        help='The minimum number of positive tags in each item (default: 0).',
        default=0
    )
    parser.add_argument(
        '--no-cuda',
        action='store_true'
    )
    parser.add_argument(
        '--gpu-id',
        type=int,
        help='The ID of selected GPU, --no-cuda must be disabled, (default: 0).',
        default=0
    )

    opt = parser.parse_args()

    # Check configuration
    if opt.num_workers == -1:
        opt.num_workers = mp.cpu_count()

    opt.device = 'cuda:{0}'.format(opt.gpu_id) if not opt.no_cuda else 'cpu'
    if not opt.no_cuda:
        assert torch.cuda.is_available()

    return opt


class TensorBoardWriter(object):
    def __init__(self, log_dir: str, purge_step: int = 0):
        self.log_dir = log_dir
        self.purge_step = purge_step

    def __enter__(self):
        self.writer = SummaryWriter(
            log_dir=self.log_dir,
            purge_step=self.purge_step
        )

        return self.writer

    def __exit__(self, type, value, traceback):
        self.writer.close()


class Trainer(object):
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        opt: argparse.Namespace
    ):
        self.model = model
        self.optimizer = optimizer
        self.opt = opt

        self.start_from_epoch = self.opt.start_from_epoch
        self.stop_at_epoch = self.opt.start_from_epoch + self.opt.epochs
        self.log_dir = './runs/{0}'.format(
            "_".join(self.opt.save_dir.split("/")[-1].split(".")))

        self.criterion = {
            'both': BCEDiceLoss(beta=1.0),
            'tag': BCEDiceLoss(beta=1.0),
            'img': BCEDiceLoss(beta=1.0)
        }

        # Initialize monitoring params
        self.best_val_loss = np.inf
        self.best_val_f1 = 0
        self.best_val_precision = 0
        self.best_val_recall = 0
        self.alpha = 0.9  # Mean over 10 iters

        self.fk_eval = PrecisionRecallFk(
            enable_logger=False, threshold=self.opt.threshold)

        if self.opt.use_rop_scheduler:
            self.rop_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=self.optimizer,
                factor=self.opt.rop_factor,
                patience=self.opt.rop_patience,
                min_lr=1e-7,
                verbose=True
            )
        elif self.opt.use_steplr_scheduler:
            self.steplr_scheduler = optim.lr_scheduler.StepLR(
                optimizer=self.optimizer,
                step_size=1,
                gamma=self.opt.sl_gamma
            )

        if self.opt.checkpoint_path is not None:
            self._load_checkpoint()

    def fit(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader
    ):
        with TensorBoardWriter(self.log_dir, purge_step=self.start_from_epoch) as writer:
            if self.opt.log_graph:
                self._log_graph(train_dataloader, writer)

            print('\nTraining model...')
            for epoch in range(self.start_from_epoch, self.stop_at_epoch):
                self._fit_an_epoch(
                    train_dataloader, val_dataloader, writer, epoch)

    def _log_graph(self, dataloader, writer):
        image_batch, tags_batch, mask_batch, _ = next(
            iter(dataloader))
        image_batch = image_batch.to(self.opt.device)
        tags_batch = tags_batch.to(self.opt.device)
        mask_batch = mask_batch.to(self.opt.device)
        writer.add_graph(self.model, (tags_batch, image_batch, mask_batch))

    def _load_checkpoint(self):
        assert os.path.isfile(self.opt.checkpoint_path)

        print('\nLoading checkpoint...')
        states = torch.load(self.opt.checkpoint_path,
                            map_location=lambda storage, loc: storage)
        print('|`-- Loading model...')
        print('+--------------------')
        model_dict = self.model.state_dict()
        excluding_layers = [
            'img_linear.weight',
            'img_linear.bias',
            'tag_linear.weight',
            'tag_linear.bias',
            'gating.linear_1.weight',
            'gating.linear_1.bias',
            'gating.linear_2.weight',
            'gating.linear_2.bias'
        ] if self.opt.exclude_top else []
        pretrained_dict = {k: v for k, v in states['model'].items()
                           if k in model_dict and k not in excluding_layers}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)
        if not self.opt.load_weights_only:
            print('|`-- Loading optimizer...')
            self.optimizer.load_state_dict(states['optimizer'])
            print('|`-- Loading best val loss...')
            self.best_val_loss = states['best_val_loss']
            print('|`-- Loading best val f1...')
            self.best_val_f1 = states['best_val_f1']
            print('|`-- Loading best val precision...')
            self.best_val_precision = states['best_val_precision']
            print(' `-- Loading best val recall...')
            self.best_val_recall = states['best_val_recall']

    def _save_checkpoint(
        self,
        new_loss,
        new_f1,
        new_precision,
        new_recall,
        epoch
    ):
        found_better_val_loss = new_loss < self.best_val_loss
        found_better_val_f1 = new_f1 > self.best_val_f1

        self.best_val_loss = np.minimum(
            self.best_val_loss, new_loss)
        self.best_val_f1 = np.maximum(
            self.best_val_f1, new_f1)
        self.best_val_precision = np.maximum(
            self.best_val_precision, new_precision)
        self.best_val_recall = np.maximum(
            self.best_val_recall, new_recall)

        states = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_f1': self.best_val_f1,
            'best_val_precision': self.best_val_precision,
            'best_val_recall': self.best_val_recall,
            'config': vars(self.opt)
        }

        if self.opt.save_best_loss and found_better_val_loss:
            print('    \__ Found a better checkpoint based on val loss -> Saving...')
            torch.save(states, os.path.join(
                self.opt.save_dir, 'best_loss.pth'))

        if self.opt.save_best_f1 and found_better_val_f1:
            print('    \__ Found a better checkpoint based on val F1 -> Saving...')
            torch.save(states, os.path.join(self.opt.save_dir, 'best_f1.pth'))

        if self.opt.save_latest:
            torch.save(states, os.path.join(self.opt.save_dir, 'latest.pth'))

        if self.opt.save_all_epochs:
            torch.save(states, os.path.join(
                self.opt.save_dir, 'epoch_{0}.pth'.format(epoch+1)))

    def _compute_running_precision_recall_f1(
        self,
        pred,
        label,
        running_precision,
        running_recall,
        running_f1
    ):
        fk_eval_dict = self.fk_eval(pred, label, betas=[1])
        running_precision = moving_avg(
            running_precision, np.nan_to_num(fk_eval_dict['precision']), self.alpha)
        running_recall = moving_avg(
            running_recall, np.nan_to_num(fk_eval_dict['recall']), self.alpha)
        running_f1 = moving_avg(
            running_f1, np.nan_to_num(fk_eval_dict['f_score']['F1']), self.alpha)

        return running_precision, running_recall, running_f1

    def _compute_batch_precision_recall_f1(
        self,
        pred,
        label,
        batch_val_idx,
        local_batch_size,
        batch_val_precision,
        batch_val_recall,
        batch_val_f1
    ):
        fk_eval_dict = self.fk_eval(pred, label, betas=[1])
        batch_val_precision[batch_val_idx] = np.nan_to_num(
            fk_eval_dict['precision']) * local_batch_size
        batch_val_recall[batch_val_idx] = np.nan_to_num(
            fk_eval_dict['recall']) * local_batch_size
        batch_val_f1[batch_val_idx] = np.nan_to_num(
            fk_eval_dict['f_score']['F1']) * local_batch_size

        return batch_val_precision, batch_val_recall, batch_val_f1

    def _fit_an_epoch(self, train_dataloader, val_dataloader, writer, epoch):
        # Training process
        self.model.train()

        # Initialize a dictionary to store numeric values
        running = {
            'loss': {
                'both': 0,
                'tag': 0,
                'img': 0,
                'sum': 0
            },
            'f1': {
                'both': 0,
                'tag': 0,
                'img': 0
            },
            'precision': {
                'both': 0,
                'tag': 0,
                'img': 0
            },
            'recall': {
                'both': 0,
                'tag': 0,
                'img': 0
            },
            'weight': {
                'tag': 0,
                'img': 0
            }
        }

        train_pbar = tqdm(train_dataloader)
        train_pbar.desc = '* Epoch {0}'.format(epoch+1)

        for batch_idx, (image_batch, tags_batch, mask_batch, label_batch) in enumerate(train_pbar):
            image_batch = image_batch.to(self.opt.device)
            tags_batch = tags_batch.to(self.opt.device)
            label_batch = label_batch.to(self.opt.device)
            mask_batch = mask_batch.to(self.opt.device)

            preds = dict()
            weight = dict()
            preds['both'], preds['tag'], preds['img'], weight['tag'], weight['img'] = \
                self.model(tags_batch, image_batch, mask_batch)

            for key in preds.keys():
                preds[key] = preds[key].masked_fill(
                    mask_batch,
                    0.0
                )

            loss = dict()
            for key in preds.keys():
                loss[key] = self.criterion[key](preds[key], label_batch)
            loss['sum'] = loss['both'] + loss['tag'] + loss['img']

            self.optimizer.zero_grad()
            loss['sum'].backward()
            self.optimizer.step()

            processed_label = label_batch.detach().cpu().numpy().astype(np.uint8)
            processed_pred = dict()
            for key in preds.keys():
                processed_pred[key] = preds[key].detach().cpu().numpy()

            # Compute running losses
            for key in running['loss'].keys():
                running['loss'][key] = moving_avg(
                    running['loss'][key], loss[key].item(), self.alpha)

            # Compute running weights
            for key in running['weight'].keys():
                running['weight'][key] = moving_avg(
                    running['weight'][key], weight[key].mean().item(), self.alpha)

            # Compute running precision, recall and f1
            for key in processed_pred.keys():
                running['precision'][key], running['recall'][key], running['f1'][key] = \
                    self._compute_running_precision_recall_f1(
                        processed_pred[key],
                        processed_label,
                        running['precision'][key],
                        running['recall'][key],
                        running['f1'][key]
                )

            train_pbar.set_postfix({
                'loss': running['loss']['both'],
                'f1': running['f1']['both'],
                'prec': running['precision']['both'],
                'recall': running['recall']['both'],
            })

        # Log to TensorBoard
        for key in running.keys():
            for subkey in running[key]:
                writer.add_scalar('{0}/train_{1}'.format(key, subkey),
                                  running[key][subkey], epoch)

        # Validation process
        self.model.eval()

        with torch.no_grad():
            # Initialize a dictionary to store 1d arrays
            batch_val = {
                'loss': {
                    'both': np.zeros(len(val_dataloader)),
                    'tag': np.zeros(len(val_dataloader)),
                    'img': np.zeros(len(val_dataloader)),
                    'sum': np.zeros(len(val_dataloader))
                },
                'f1': {
                    'both': np.zeros(len(val_dataloader)),
                    'tag': np.zeros(len(val_dataloader)),
                    'img': np.zeros(len(val_dataloader))
                },
                'precision': {
                    'both': np.zeros(len(val_dataloader)),
                    'tag': np.zeros(len(val_dataloader)),
                    'img': np.zeros(len(val_dataloader))
                },
                'recall': {
                    'both': np.zeros(len(val_dataloader)),
                    'tag': np.zeros(len(val_dataloader)),
                    'img': np.zeros(len(val_dataloader))
                },
                'weight': {
                    'tag': np.zeros(len(val_dataloader)),
                    'img': np.zeros(len(val_dataloader))
                },
            }

            if self.opt.log_weight_hist:
                all_val_weights = {
                    'tag': [],
                    'img': []
                }

            num_items = 0

            val_pbar = tqdm(val_dataloader)
            val_pbar.desc = '\__ Validating'

            for batch_val_idx, (image_batch, tags_batch, mask_batch, label_batch) in enumerate(val_pbar):
                image_batch = image_batch.to(self.opt.device)
                tags_batch = tags_batch.to(self.opt.device)
                label_batch = label_batch.to(self.opt.device)
                mask_batch = mask_batch.to(self.opt.device)

                preds = dict()
                weight = dict()
                preds['both'], preds['tag'], preds['img'], weight['tag'], weight['img'] = \
                    self.model(tags_batch, image_batch, mask_batch)

                for key in preds.keys():
                    preds[key] = preds[key].masked_fill(
                        mask_batch,
                        0.0
                    )

                val_loss = dict()
                for key in preds.keys():
                    val_loss[key] = self.criterion[key](
                        preds[key], label_batch)
                val_loss['sum'] = val_loss['both'] + \
                    val_loss['tag'] + val_loss['img']

                processed_label = label_batch.detach().cpu().numpy().astype(np.uint8)
                processed_pred = dict()
                for key in preds.keys():
                    processed_pred[key] = preds[key].detach().cpu().numpy()

                local_batch_size = label_batch.size(0)
                num_items += local_batch_size

                # Compute sum batch losses
                for key in batch_val['loss'].keys():
                    batch_val['loss'][key][batch_val_idx] = val_loss[key].item(
                    ) * local_batch_size

                # Compute sum batch weights
                for key in batch_val['weight'].keys():
                    batch_val['weight'][key][batch_val_idx] = weight[key].mean().item(
                    ) * local_batch_size

                # Keep all available weight values (if needed)
                if self.opt.log_weight_hist:
                    for key in batch_val['weight'].keys():
                        all_val_weights[key].extend(
                            weight[key].detach().cpu().numpy().reshape(-1))

                # Compute sum batch precision, recall and f1
                for key in processed_pred.keys():
                    batch_val['precision'][key], batch_val['recall'][key], batch_val['f1'][key] = \
                        self._compute_batch_precision_recall_f1(
                            processed_pred[key],
                            processed_label,
                            batch_val_idx,
                            local_batch_size,
                            batch_val['precision'][key],
                            batch_val['recall'][key],
                            batch_val['f1'][key]
                    )

                val_pbar.set_postfix({
                    'loss': batch_val['loss']['both'][batch_val_idx] / local_batch_size,
                    'f1': batch_val['f1']['both'][batch_val_idx] / local_batch_size,
                    'prec': batch_val['precision']['both'][batch_val_idx] / local_batch_size,
                    'recall': batch_val['recall']['both'][batch_val_idx] / local_batch_size,
                })

            mean_val = dict()
            for key in batch_val.keys():
                mean_val[key] = dict()
                for subkey in batch_val[key].keys():
                    mean_val[key][subkey] = np.sum(
                        batch_val[key][subkey]) / num_items

            if self.opt.use_rop_scheduler:
                self.rop_scheduler.step(mean_val['loss']['sum'])
            elif self.opt.use_steplr_scheduler:
                self.steplr_scheduler.step()

            # Log to TensorBoard
            for key in mean_val.keys():
                for subkey in mean_val[key].keys():
                    writer.add_scalar('{0}/val_{1}'.format(key, subkey),
                                      mean_val[key][subkey], epoch)
            if self.opt.log_weight_hist:
                for key in all_val_weights.keys():
                    writer.add_histogram(
                        'weight/{0}'.format(key), np.array(all_val_weights[key]), epoch)

            # Save checkpoint
            self._save_checkpoint(
                new_loss=mean_val['loss']['both'],
                new_f1=mean_val['f1']['both'],
                new_precision=mean_val['precision']['both'],
                new_recall=mean_val['recall']['both'],
                epoch=epoch
            )
