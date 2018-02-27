import tensorflow as tf

def preprocessing_op(image_op, config):
    with tf.name_scope("preprocessing"):
        # split input
        rgb, dep, seg = image_op
        
        # reshape and concat
        rgb = tf.reshape(rgb, (-1, config['img_height'], config['img_width'], config['img_num_channels']))
        seg = tf.reshape(seg, (-1, config['img_height'], config['img_width'], config['img_num_channels']))
        seg = tf.expand_dims(seg[:, :, :, 0], 3)
        dep = tf.reshape(dep, (-1, config['img_height'], config['img_width'], 1))
        x = tf.concat([rgb, seg, dep], axis=3)

        max_seq_len = tf.shape(x)[0]
        num_channels = 5

        # random rotatioan
        angle = tf.tile(tf.random_uniform([1],minval=-0.05, maxval=0.05, dtype=tf.float32), 
                           tf.expand_dims(max_seq_len, axis=0))
        x = tf.contrib.image.rotate(x, angle)

        # random crop
        crop_amount = tf.random_uniform([], maxval=tf.int32.max, dtype=tf.int32) % 5 + 2
        crop_offset_y = tf.random_uniform([], maxval=tf.int32.max, dtype=tf.int32) % (crop_amount + 1)
        crop_offset_x = tf.random_uniform([], maxval=tf.int32.max, dtype=tf.int32) % (crop_amount + 1)
        crop_offset = tf.stack([ 0, crop_offset_y, crop_offset_x, 0 ])
        crop_size = tf.stack([ 
            max_seq_len, 
            config['img_height'] - crop_amount, 
            config['img_width'] - crop_amount, 
            num_channels
        ])
        x = tf.slice(x, crop_offset, crop_size)
        x = tf.image.resize_images(x, [config['img_height'], config['img_width']], 
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        x = tf.reshape(x, (-1, config['img_height'], config['img_width'], num_channels))

        # random flip
        do_flip = tf.random_uniform([]) > .5
        x = tf.cond(do_flip, lambda: tf.reverse(x, [2]), lambda: x)

        # convert to float in range [0, 1]
        x = tf.cast(x, tf.float32) / 255.

        # apply image adjustments to rgb channels
        rgb = tf.slice(x, [0,0,0,0], [-1,-1,-1,3])
        rgb = tf.image.random_contrast(rgb, lower=.8, upper=1.2)
        rgb = tf.image.random_brightness(rgb, max_delta=.1)

        # rejoin channels
        x = tf.concat([rgb, tf.slice(x, [0,0,0,3], [-1,-1,-1,-1])], axis=3)

        # random channel shift
        x = x + tf.random_uniform([num_channels], minval=-.1, maxval=.1, dtype=tf.float32)

        # make sure values are still in [0, 1]
        x = tf.minimum(tf.maximum(x, 0.), 1.)

        # zero mean
        x = x * 2. - 1.

        return x

def read_and_decode_sequence(filename_queue, config):
    # Create a TFRecordReader.
    readerOptions = tf.python_io.TFRecordOptions(compression_type=tf.python_io.TFRecordCompressionType.GZIP)
    reader = tf.TFRecordReader(options=readerOptions)
    _, serialized_example = reader.read(filename_queue)

    # Read one sequence sample.
    # The training and validation files contains the following fields:
    # - label: label of the sequence which take values between 1 and 20.
    # - length: length of the sequence, i.e., number of frames.
    # - depth: sequence of depth images. [length x height x width x numChannels]
    # - rgb: sequence of rgb images. [length x height x width x numChannels]
    # - segmentation: sequence of segmentation maskes. [length x height x width x numChannels]
    # - skeleton: sequence of flattened skeleton joint positions. [length x numJoints]
    #
    # The test files doesn't contain "label" field.
    # [height, width, numChannels] = [80, 80, 3]
    with tf.name_scope("TFRecordDecoding"):
        context_encoded, sequence_encoded = tf.parse_single_sequence_example(
                serialized_example,
                # "label" and "lenght" are encoded as context features.
                context_features={
                    "label": tf.FixedLenFeature([], dtype=tf.int64),
                    "length": tf.FixedLenFeature([], dtype=tf.int64)
                },
                # "depth", "rgb", "segmentation", "skeleton" are encoded as sequence features.
                sequence_features={
                    "depth": tf.FixedLenSequenceFeature([], dtype=tf.string),
                    "rgb": tf.FixedLenSequenceFeature([], dtype=tf.string),
                    "segmentation": tf.FixedLenSequenceFeature([], dtype=tf.string),
                    "skeleton": tf.FixedLenSequenceFeature([], dtype=tf.string),
                })


        # Fetch required data fields.
        seq_rgb = tf.decode_raw(sequence_encoded['rgb'], tf.uint8)
        seq_dep = tf.decode_raw(sequence_encoded['depth'], tf.uint8)
        seq_seg = tf.decode_raw(sequence_encoded['segmentation'], tf.uint8)

        seq_label = context_encoded['label']
        # Tensorflow requires the labels start from 0. Before you create submission csv,
        # increment the predictions by 1.
        seq_label = seq_label - 1
        seq_len = tf.to_int32(context_encoded['length'])
        # Output dimnesionality: [seq_len, height, width, numChannels]
        seq_all = [seq_rgb, seq_dep, seq_seg]
        seq_all = preprocessing_op(seq_all, config)

        return [seq_all, seq_label, seq_len]

def read_and_decode_sequence_test_data(filename_queue, config):
    """
    Replace label field with id field because test data doesn't contain labels.
    """
    # Create a TFRecordReader.
    readerOptions = tf.python_io.TFRecordOptions(compression_type=tf.python_io.TFRecordCompressionType.GZIP)
    reader = tf.TFRecordReader(options=readerOptions)
    _, serialized_example = reader.read(filename_queue)

    # Read one sequence sample.
    # The training and validation files contains the following fields:
    # - label: label of the sequence which take values between 1 and 20.
    # - length: length of the sequence, i.e., number of frames.
    # - depth: sequence of depth images. [length x height x width x numChannels]
    # - rgb: sequence of rgb images. [length x height x width x numChannels]
    # - segmentation: sequence of segmentation maskes. [length x height x width x numChannels]
    # - skeleton: sequence of flattened skeleton joint positions. [length x numJoints]
    #
    # The test files doesn't contain "label" field.
    # [height, width, numChannels] = [80, 80, 3]
    with tf.name_scope("TFRecordDecoding"):
        context_encoded, sequence_encoded = tf.parse_single_sequence_example(
                serialized_example,
                # "label" and "lenght" are encoded as context features.
                context_features={
                    "id": tf.FixedLenFeature([], dtype=tf.int64),
                    "length": tf.FixedLenFeature([], dtype=tf.int64)
                },
                # "depth", "rgb", "segmentation", "skeleton" are encoded as sequence features.
                sequence_features={
                    "depth": tf.FixedLenSequenceFeature([], dtype=tf.string),
                    "rgb": tf.FixedLenSequenceFeature([], dtype=tf.string),
                    "segmentation": tf.FixedLenSequenceFeature([], dtype=tf.string),
                    "skeleton": tf.FixedLenSequenceFeature([], dtype=tf.string),
                })


        # Fetch required data fields.
        seq_rgb = tf.decode_raw(sequence_encoded['rgb'], tf.uint8)
        seq_dep = tf.decode_raw(sequence_encoded['depth'], tf.uint8)
        seq_seg = tf.decode_raw(sequence_encoded['segmentation'], tf.uint8)
        seq_len = tf.to_int32(context_encoded['length'])
        # Output dimnesionality: [seq_len, height, width, numChannels]
        seq_all = [seq_rgb, seq_dep, seq_seg]
        seq_all = preprocessing_op(seq_all, config)
        seq_id = context_encoded['id']

        return [seq_all, seq_id, seq_len]


def input_pipeline(filenames, config, name='input_pipeline', shuffle=True, mode='training'):
    with tf.name_scope(name):
        # Read the data from TFRecord files, decode and create a list of data samples by using threads.
        if mode is "training":
            # Create a queue of TFRecord input files.
            filename_queue = tf.train.string_input_producer(filenames, num_epochs=config['num_epochs'], shuffle=shuffle)
            sample_list = [read_and_decode_sequence(filename_queue, config) for _ in range(config['ip_num_read_threads'])]
            batch_rgb, batch_labels, batch_lens = tf.train.batch_join(sample_list,
                                                    batch_size=config['batch_size'],
                                                    capacity=config['ip_queue_capacity'],
                                                    enqueue_many=False,
                                                    dynamic_pad=True,
                                                    allow_smaller_final_batch = False,
                                                    name="batch_join_and_pad")
            return batch_rgb, batch_labels, batch_lens

        else:
            filename_queue = tf.train.string_input_producer(filenames, num_epochs=1, shuffle=False)
            sample_list = [read_and_decode_sequence_test_data(filename_queue, config) for _ in range(config['ip_num_read_threads'])]
            batch_rgb, batch_ids, batch_lens = tf.train.batch_join(sample_list,
                                                    batch_size=config['batch_size'],
                                                    capacity=config['ip_queue_capacity'],
                                                    enqueue_many=False,
                                                    dynamic_pad=True,
                                                    allow_smaller_final_batch = True,
                                                    name="batch_join_and_pad")
            return batch_rgb, batch_ids, batch_lens
