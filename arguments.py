import argparse
import sys

from configs import EXPERIMENT_CONFIGS, apply_preset, explicit_cli_options

def arg_parse():
    parser = argparse.ArgumentParser(description='PyTorch TKDE_label_noise Training')
    parser.add_argument('--preset', type=str, choices=sorted(EXPERIMENT_CONFIGS),
                        default=None, help='Experiment preset name')
    parser.add_argument('--dataset', type=str, default='dblp', help='Data set')


    parser.add_argument("--label_rate", type=float, default=0.01, 
                        help='rate of labeled data')
    parser.add_argument('--ptb_rate', type=float, default=0.2, 
                        help="noise ptb_rate")
    parser.add_argument('--noise', type=str, default='uniform', choices=['uniform', 'pair'], 
                        help='type of noises')
    
    parser.add_argument("--weight_all", type=float, default=1, 
                        help='weight of combined decision loss')
    parser.add_argument("--weight_con", type=float, default=1, 
                        help='weight of consistent loss')
    parser.add_argument("--weight_pse", type=float, default=1, 
                        help='rate of pseudo label loss')

    parser.add_argument('--lambda-epochs', type=int, default=50, metavar='N',
                        help='gradually increase the value of lambda from 0 to 1')
    parser.add_argument('--views', type=int, default=3, 
                        help='The number of multi-view')
    parser.add_argument('--cuda', type=int, default=True, 
                        help='')

    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001, 
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden1', type=int, default=256,
                        help='Number of hidden units.')
    parser.add_argument('--hidden', type=int, default=256,
                        help='Number of hidden units.')
    parser.add_argument('--droprate', type=float, default=0.5,
                        help='Dropout rate of the layer (1 - keep probability).')
    parser.add_argument('--patience', type=int, default=100, help='Patience')
    parser.add_argument('--T', default=0.5, type=float,
                        help='pseudo label temperature')
    parser.add_argument('--threshold', default=0.95, type=float,
                        help='pseudo label threshold')

    parser.add_argument('--K', type=int, default=5)
    parser.add_argument('--normalize_features', type=bool, default=True) # True
    parser.add_argument('--random_splits', type=bool, default=False)

    args = parser.parse_args()
    return apply_preset(args, explicit_cli_options(sys.argv))