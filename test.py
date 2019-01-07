import tensorflow as tf

# text = tf.constant(['yes', 'you', 'know'], dtype=tf.string)
#
# chars = tf.string_split(text, delimiter='')
#
# # labels_sp = tf.SparseTensor(
# #     chars.indices,
# #     self._char_to_label_table.lookup(chars.values),
# #     chars.dense_shape
# # )
# #
# # if return_dense:
# #     labels = tf.sparse_tensor_to_dense(labels_sp, default_value=pad_value)
# # else:
# #     labels = labels_sp
# #
# # if return_lengths:
# #     text_lengths = tf.sparse_reduce_sum(
# #         tf.SparseTensor(
# #             chars.indices,
# #
# #             chars.dense_shape
# #         ),
# #         axis=1
# #     )
# #     text_lengths.set_shape([None])
#
#
# res = tf.fill([tf.shape(chars.indices)[0]], 1)
#
#
# with tf.Session() as sess:
#     print(sess.run(res))

# output file name string to a queue
filename_queue = tf.train.string_input_producer(['/data/ocr/Synthetic_Chinese_String_Dataset/chinese_v1.tfrecord'], num_epochs=None)
# create a reader from file queue
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
# get feature from serialized example

features = tf.parse_single_example(serialized_example,
        features={
            'a': tf.FixedLenFeature([], tf.float32),
            'b': tf.FixedLenFeature([2], tf.int64),
            'c': tf.FixedLenFeature([], tf.string)
        }
    )


with tf.Session() as sess:
    print(sess.run(features))









