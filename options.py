import argparse

def options():
    parser = argparse.ArgumentParser()

    parser.add_argument('--datapath', type=str, default='/sequoia/data2/teboli/irc_nonblind/data/training_uniform')
    parser.add_argument('--n_in', type=int, default=2, help='CPCR iterations; S in the paper.')
    parser.add_argument('--n_out', type=int, default=5, help='HQS iterations; T in the paper.')
    parser.add_argument('--n_epochs', type=int, default=20, help='Training epochs.')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--datasize', type=int, default=3000, help='Number of training samples.')
    parser.add_argument('--ps', type=int, default=180, help='Size of training images.')
    parser.add_argument('--load_epoch', type=int, default=0, help='Epoch to relead training; 0 is no loading.')
    parser.add_argument('--noise_level', type=float, default=12.75)
    parser.add_argument('--blind', type=int, default=1, help='Training on all noise levels between 0.5 and args.noise_level or only on args.noise_level')
    parser.add_argument('--datapath', type=str, default='./data/training_uniform', help='Path of training samples.')
    parser.add_argument('--lambd', type=float, default=1e-2, help='Weight for regularization in Eq.(10).')

    return parser
