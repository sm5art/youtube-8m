import tensorflow as tf

def createTargetVec(labels):
    out = np.zeros((1, 4716))
    for label in labels:
        out[0,label] = 1
    return out

class DataContainer:
    def __init__(self, filename, batch_size):
        self.batch_size = batch_size
        self.vframes = []
        self.aframes = []
        self.labels = []
        i = 0
        for record in tf.python_io.tf_record_iterator(filename):
            if i>15:
                break
            record = tf.train.SequenceExample.FromString(record)
            n_frames = len(record.feature_lists.feature_list['audio'].feature)
            vframe = []
            aframe = []
            sess = tf.InteractiveSession()
            labels = record.context.feature['labels'].int64_list.value
            self.labels.append(labels)
            for i in range(n_frames):
                vframe.append(tf.cast(tf.decode_raw(
                        record.feature_lists.feature_list['rgb'].feature[i].bytes_list.value[0],tf.uint8)
                               ,tf.float32).eval())
                aframe.append(tf.cast(tf.decode_raw(
                        record.feature_lists.feature_list['audio'].feature[i].bytes_list.value[0],tf.uint8)
                               ,tf.float32).eval())
            sess.close()
            self.vframes.append(vframe)
            self.aframes.append(aframe)
            i += 1

    def get_training_batch(self):
        batch_v = self.vframes[-self.batch_size:]
        del self.vframes[-self.batch_size:]
        batch_a = self.aframes[-self.batch_size:]
        del self.aframes[-self.batch_size:]
        labels = self.labels[-self.batch_size:]
        del self.labels[-self.batch_size:]
        final = []
        for i in range(batch_v):
            final.append(([batch_v[i], batch_a[i]], labels))
        return final
