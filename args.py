import argparse

def get_processing_args():
    """Get arguments needed in process.py."""

    parser = argparse.ArgumentParser('Preprocess dataset.')
    
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
                        default='None',
                        help= "Choose resampling strategy from UderSample, OverSample or None")

    parser.add_argument('--max_seq_len',
                       type=int,
                       default=50)

    parser.add_argument('--train_vecs',
                        type=str,
                        default='./data/train.npz')

    parser.add_argument('--test_vecs',
                        type=str,
                        default='./data/test.npz')

    parser.add_argument('--train_test_split',
                        type=float,
                        default=0.2)

    parser.add_argument('--vocab_path',
                        type=str,
                        default='./data/vocab.json')

    args = parser.parse_args()

    return args

