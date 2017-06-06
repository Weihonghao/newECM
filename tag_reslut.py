from emotion_predictor import EmotionPredictor
import tensorflow as tf

answer_file_name = "answer.txt"
tag_file_name = "tag.txt"

tf.python.control_flow_ops = tf
model = EmotionPredictor(classification='ekman', setting='mc')

batch_size = 100

answer_file = open(answer_file_name,'r')
tag_file = open(tag_file_name,'w')
i = 1
line_batch = []
with open(answer_file_name) as file:
    for line in file:
        line = line.lower()
        line_batch.append(line)
        if (i % batch_size) == 0:
            indices = model._tweet_to_indices(line_batch)
            predictions = model.model.predict(indices, verbose=False)
            tag_result = predictions.argmax(axis=-1)
            for j, result in enumerate(tag_result):
            	tag_file.write(str(tag_result[j]) + "\n")
            line_batch = []
        i += 1
        print(i)

tag_file.close()
answer_file.close()
