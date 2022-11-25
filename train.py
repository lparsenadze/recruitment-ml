from args import get_training_args
from models import BiLSTM, get_logger
import tensorflow as tf
import numpy as np
from settings import *
from time import sleep
from tqdm import tqdm
import json
import tensorflow_addons as tfa
import datetime


@tf.function
def train_step(data, labels):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(data, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    #print(predictions.shape)
    #print(labels.shape)
    #shape = (labels.shape[0]*labels.shape[1], labels.shape[-1])
    #labels = tf.reshape(labels, shape=shape)
    #predictions = tf.reshape(predictions, shape=shape)
     
    train_loss(loss)
    train_accuracy(tf.reshape(labels, (-1, 1)), predictions)

@tf.function
def test_step(data, labels):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(data, training=False)
    t_loss = loss_object(labels, predictions)
    
    #shape = (labels.shape[0]*labels.shape[1], labels.shape[-1])
    test_loss(t_loss)
    test_accuracy(tf.reshape(labels, (-1,1)), predictions)

def load_vecs(path):
    ds = np.load(path)
    return ds['X'], ds['y']


if __name__ == '__main__':
    args_ = get_training_args()
    logger = get_logger(filename=TRAINING_LOG_PATH)
    
    logger.info('Loading data...')
    X_train, y_train = load_vecs(args_.train_vecs) 
    X_test, y_test = load_vecs(args_.test_vecs)
    train_ds = tf.data.Dataset.from_tensor_slices(
        (X_train, y_train)).shuffle(10000).batch(args_.batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices(
        (X_test, y_test)).shuffle(10000).batch(args_.batch_size)
    with open(args_.vocab_path, 'r') as f:
        vocab = json.load(f)
    logger.info('Completed.\n') 
    
    logger.info('Loading model...')
    n_classes = 1
    model = BiLSTM(len(vocab), args_.embed_dim, args_.max_seq_len, args_.dense_dims,
                   n_classes, num_lstm_units=args_.num_lstm_units, num_bilstm_layers=args_.num_bilstm_layers)
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.Adam()
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    #train_accuracy = tfa.metrics.F1Score(num_classes=n_classes, average="macro")
    train_accuracy = tf.keras.metrics.BinaryAccuracy(name='binary_accuracy', dtype=None, threshold=0.5)

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    #test_accuracy = tfa.metrics.F1Score(num_classes=n_classes, average="macro")
    test_accuracy = tf.keras.metrics.BinaryAccuracy(name='binary_accuracy', dtype=None, threshold=0.5)
    logger.info("Completed.\n")
    
    train_summary_writer = tf.summary.create_file_writer('./logs/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    test_summary_writer = tf.summary.create_file_writer('./logs/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    logger.info('Starting training...')
    for epoch in range(args_.num_epochs):
        with tqdm(enumerate(train_ds), unit="batch") as tepoch:
            #embed()
            tepoch.set_description(f"Epoch {epoch}")

            # Reset the metrics at the start of the next epoch
            train_loss.reset_states()
            train_accuracy.reset_states()

            for batch, (data, labels) in tepoch:
                train_step(data, labels)
                with train_summary_writer.as_default():
                    tf.summary.scalar('Train Loss', train_loss.result(), step=epoch * len(train_ds) + batch)
                    tf.summary.scalar('Train Accuracy', train_accuracy.result(), step=epoch * len(train_ds) + batch)

            #for test_data, test_labels in test_ds:
            #    test_step(test_data, test_labels)
        
        
                tepoch.set_postfix(
                    train_LOSS=train_loss.result().numpy(), 
                    train_ACC=100. * train_accuracy.result().numpy()) 
                    #test_loss=test_loss.result().numpy(), 
                    #test_f1_score=100. * test_accuracy.result().numpy())
                sleep(0.1)

        with tqdm(enumerate(test_ds), unit="batch") as tepoch:
            #embed()
            tepoch.set_description(f"Test {epoch}")

            # Reset the metrics at the start of the next epoch
            test_loss.reset_states()
            test_accuracy.reset_states()

            for batch, (data, labels) in tepoch:
                test_step(data, labels)
                with train_summary_writer.as_default():
                    tf.summary.scalar('Test Loss', test_loss.result(), step=epoch * len(test_ds) + batch)
                    tf.summary.scalar('Test Accuracy', test_accuracy.result(), step=epoch * len(test_ds) + batch)
            #for test_data, test_labels in test_ds:
            #    test_step(test_data, test_labels)


                tepoch.set_postfix(
                    #train_loss=train_loss.result().numpy(),
                    #train_f1_score=100. * train_accuracy.result().numpy())
                    test_LOSS=test_loss.result().numpy(), 
                    test_ACC=100. * test_accuracy.result().numpy())
                sleep(0.1)
    model.save(args_.model_path)
    logger.info("Done.\n")
