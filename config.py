import argparse
from pathlib import Path


def get_config():
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description='CaDH: dual-perspective credibility learning and uncertainty-aware similarity modulation')
    p.add_argument('--dataset', choices=['flickr','coco','nuswide','imagenet'], default='flickr')
    p.add_argument('--n-bits', type=int, choices=[16,32,64,128], default=16)
    p.add_argument('--backbone', choices=['resnet50','vit_b_16','tiny'], default='resnet50')
    p.add_argument('--data-dir', default=str(root.parent/'_datasets'))
    p.add_argument('--output-root', default=str(root/'output'))
    p.add_argument('--n-epochs', type=int, default=100)
    p.add_argument('--eval-interval', type=int, default=5)
    p.add_argument('--early-stop-patience', type=int, default=0, help='0 disables; otherwise number of validations without improvement')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--n-workers', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='auto')
    p.add_argument('--backbone-lr', type=float, default=1e-5)
    p.add_argument('--head-lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--lambda-fuzzy', type=float, default=1.)
    p.add_argument('--lambda-quantization', type=float, default=.1)
    p.add_argument('--margin', type=float, default=.25)
    p.add_argument('--tau', type=float, default=9.)
    p.add_argument('--gamma', type=float, default=0.)
    p.add_argument('--mode', choices=['ssm','cosine'], default='ssm')
    p.add_argument('--no-pretrained', action='store_true')
    p.add_argument('--synthetic', action='store_true', help='Plumbing checks only; never scientific results')
    args = p.parse_args()
    if min(args.n_epochs,args.eval_interval,args.batch_size) < 1 or args.n_workers < 0 or args.early_stop_patience < 0:
        p.error('Invalid epoch, interval, batch size, workers or patience')
    if args.backbone == 'tiny' and not args.synthetic:
        p.error('tiny backbone is for synthetic checks only')
    return vars(args)
