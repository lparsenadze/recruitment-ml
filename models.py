import logging
import tensorflow as tf

class BiLSTM(tf.keras.Model):
    def __init__(self, 
                 vocab_size, 
                 embed_dim, 
                 max_seq_len, 
                 dense_dims, 
                 n_classes=1, 
                 num_lstm_units=50, 
                 num_bilstm_layers=1):

        super(BiLSTM, self).__init__()
        self.vocab_size = vocab_size
        self_max_seq_len = max_seq_len
        self.num_bilstm_layers = num_bilstm_layers
        self.embedding_layer = tf.keras.layers.Embedding(input_dim=vocab_size,
                            output_dim=embed_dim,
                            input_length=max_seq_len,
                            trainable=True)
        self.bilstms = []
        for layer in range(num_bilstm_layers):
            self.bilstms.append(tf.keras.layers.Bidirectional(
                                    tf.keras.layers.LSTM(units=num_lstm_units)))
        self.dense = tf.keras.layers.Dense(dense_dims, activation='relu')
        self.dropout = tf.keras.layers.Dropout(rate=0.25)
        self.out = tf.keras.layers.Dense(n_classes, activation="sigmoid")


    def call(self, x, training=True):
        out = self.embedding_layer(x)
        #for layer in range(self.num_bilstm_layers):
        out = self.bilstms[0](out)
        out = self.dense(out)
        out = self.dropout(out, training=training)
        out = self.out(out)
        return out
        

def get_logger(level=logging.INFO,
               filename=None,
               formatting="%(asctime)s %(name)-12s %(levelname)-8s %(message)s"):
    """
    Returns a basic logger.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(formatting)

    file_handler = logging.FileHandler(filename)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger
