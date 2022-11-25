import argparse

def get_processing_args():
    """Get arguments needed in process.py."""
    parser = argparse.ArgumentParser('Preprocess dataset.')
    
    add_common_args(parser)

    parser.add_argument('--logging_path',
                        type=str,
                        default='./logs/processing_logs.txt'),

    parser.add_argument('--dataset',
                        type=str,
                        default='./data/gutenberg-paragraphs.json')
    
    parser.add_argument('--method',
                        type=str,
                        default='lemm_v',
                        help= "Choose preprocessing from cased, uncased, lemm_v, lemm, or stemm")

    parser.add_argument('--strategy',
                        type=str,
                        default='UnderSample',
                        help= "Choose resampling strategy from UderSample, OverSample or None")

    parser.add_argument('--train_test_split',
                        type=float,
                        default=0.2)

    args = parser.parse_args()

    return args

def get_training_args():
    """Get arguments needed in train.py"""
    parser = argparse.ArgumentParser('Train model.')
    
    add_common_args(parser)

    parser.add_argument('--model_path',
                        type=str,
                        default='./data/model')

    parser.add_argument('--embed_dim',
                        type=int,
                        default=256)

    parser.add_argument('--num_lstm_units',
                        type=int,
                        default=64)

    parser.add_argument('--num_bilstm_layers',
                        type=int,
                        default=1)
    
    parser.add_argument('--num_epochs',
                        type=int,
                        default=3)

    parser.add_argument('--batch_size',
                        type=int,
                        default=32)

    parser.add_argument('--dense_dims',
                        type=int,
                        default=256)


    return parser.parse_args()


def add_common_args(parser):
    
    parser.add_argument('--max_seq_len',
                       type=int,
                       default=100)


    parser.add_argument('--train_vecs',
                        type=str,
                        default='./data/train.npz')

    parser.add_argument('--test_vecs',
                        type=str,
                        default='./data/test.npz')

    parser.add_argument('--vocab_path',
                        type=str,
                        default='./data/vocab.json')
