import warnings

from torch import optim

from magneto.utils import parse_train_args
from magneto.data import get_dataloaders
from magneto.model import MAGNeto
from magneto.utils import Trainer

warnings.filterwarnings("ignore")


def main():
    ##### GET CONFIGURATION #####
    opt = parse_train_args()

    ##### PREPARING DATASETS #####
    print('\nPreparing datasets...')
    train_dataloader, val_dataloader, vocab_size = get_dataloaders(
        train_csv_path=opt.train_csv_path,
        val_csv_path=opt.val_csv_path,
        vocab_path=opt.vocab_path,
        img_dir=opt.img_dir,
        tagaug_add_max_ratio=opt.tagaug_add_max_ratio,
        tagaug_drop_max_ratio=opt.tagaug_drop_max_ratio,
        train_batch_size=opt.train_batch_size,
        val_batch_size=opt.val_batch_size,
        max_len=opt.max_len,
        num_workers=opt.num_workers,
        pin_memory=True if not opt.no_cuda else False
    )

    ##### CREATE MODEL #####
    model = MAGNeto(
        d_model=opt.d_model,
        vocab_size=vocab_size,
        t_blocks=opt.t_blocks,
        t_heads=opt.t_heads,
        t_dim_feedforward=opt.t_dim_feedforward,
        i_blocks=opt.i_blocks,
        i_heads=opt.i_heads,
        i_dim_feedforward=opt.i_dim_feedforward,
        img_backbone=opt.img_backbone,
        g_dim_feedforward=opt.g_dim_feedforward,
        dropout=opt.dropout,
    )
    model = model.to(opt.device)

    ##### CREATE OPTIMIZER #####
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=opt.lr,
        momentum=0.9
    )

    ##### CREATE TRAINER AND START THE TRAINING PROCESS #####
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        opt=opt
    )
    trainer.fit(train_dataloader, val_dataloader)


if __name__ == '__main__':
    main()
