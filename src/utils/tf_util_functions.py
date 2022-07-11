from datetime import datetime
import tensorflow as tf

def tf_convert_float_to_binary_mask(mask: tf.Tensor, threshold: float = 0.5):
    return tf.where(tf.cast(mask, tf.float32) >= threshold, 1.0, 0.0)

def tf_shift_array(arr, num, fill=0):
    shifted = tf.roll(arr, num, 0)
    if num > 0:
        result = tf.concat([tf.fill([num], fill), shifted[num:]], axis=0)
    elif num < 0:
        result = tf.concat([shifted[:num], tf.fill([abs(num)], fill)], axis=0)
    else:
        result = arr
    return result

def tf_convert_mask_to_cup_format(arr: tf.Tensor) -> tf.RaggedTensor:
    # create shifted array with orignal[n+1] = shifted[n]
    shifted_arr = tf_shift_array(arr, -1, fill=arr[-1])
    
    # step_changes must differ with in value with their successor
    # begin of step arr[n] = 0, arr[n+1] = 1
    # -> step starts at n+1
    # end of step arr[n] = 1, arr[n+1] = 0
    # -> step ends at n
    bool_arr = tf.where(arr == 1.0, True, False)
    bool_shifted_arr = tf.where(shifted_arr == 1.0, True, False)
    step_changes = tf.math.logical_xor(bool_arr, bool_shifted_arr)
    
    # create array with indices and apply boolean mask for selection
    changes = tf.cast(tf.squeeze(tf.where(step_changes), axis=1), dtype=tf.int32)
    # add start of first step at 0 if the prediction starts with a step
    if bool_arr[0]:
        changes = tf.concat([tf.constant([-1]), changes], axis=0)
    # add end of last step at len(arr) - 1 if the prediction ends with a step
    if bool_arr[-1]:
        changes = tf.concat([changes, tf.constant([bool_arr.shape[0] - 1])], axis=0)
    
    # explanation: see comments above and *.ipynb
    correct_starts = tf.tile(tf.constant([1, 0]), [int(tf.shape(changes)[0] / 2)])
    changes += correct_starts
    
    # convert array (vector) to matrix
    nested = tf.reshape(changes, (int(tf.shape(changes)[0] / 2), 2))
    
    # delete all steps of length one, i. e. start == end
    nested = nested[~(nested[:, 0] == nested[:, 1])]
    return nested

@tf.function        
def sample_function(sample):
    # _, length, channels -> _, channels, length
    sampleT = tf.transpose(sample)
    steps = tf.map_fn(channel_function, sampleT, fn_output_signature=tf.RaggedTensorSpec(shape=(1, None, None), dtype=tf.int32, ragged_rank=2, row_splits_dtype=tf.int64))
    return steps

@tf.function
def tf_process_channel(mask: tf.Tensor) -> tf.RaggedTensor:
    binary_mask = tf_convert_float_to_binary_mask(mask)
    steps = tf_convert_mask_to_cup_format(binary_mask)
    return steps

@tf.function
def channel_function(channel):
    steps = tf_process_channel(channel)
    tensor = tf.ragged.stack(steps)
    return tensor

@tf.function
def tf_conversion(input_tensor):
    t = tf.map_fn(fn=sample_function, elems=input_tensor, fn_output_signature=tf.RaggedTensorSpec(shape=(2, 1, None, None), dtype=tf.int32, ragged_rank=3, row_splits_dtype=tf.int64))
    batch_size = t.shape[0]
    channels = t.shape[1]
    collector = []
    for b in range(batch_size):
        for i in range(channels):
            collector.append(t[b, i, :, :, :])
    return tf.concat(collector, 0)

def tf_printc(source: str = '[Unknown]', message: str = '', leading: str = '', **kwargs) -> None:
    '''
    prints [message] with a [src] and a timestamp attached
    '''
    tf.print(f'{leading}{datetime.now()}   {source} {message}')