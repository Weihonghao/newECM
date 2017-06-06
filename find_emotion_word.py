from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
import mmap
def whetherEmotion(word, threshold):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_result = analyzer.polarity_scores(word)
    if abs(sentiment_result['compound']) > threshold:
        return True

    return False

def get_line_number(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines
if __name__ == '__main__':
	threshold = 0.5
	out = open("/commuter/chatbot/Commuter/data/emotion_vocab.txt",'w')
	emotion_word_set = set()
	line_number = get_line_number('/commuter/chatbot/Commuter/question.txt')
	print(line_number)
	f = open("/commuter/chatbot/Commuter/question.txt",'r')
	for line in tqdm(f, total=line_number):#f.readlines():
		line = line.strip().split()
		for each in line:
			if whetherEmotion(each, threshold):
				emotion_word_set.add(each)
	f.close()

	#emotion_word_set = set()
	f = open("/commuter/chatbot/Commuter/answer.txt",'r')
	for line in tqdm(f, total=line_number):#f.readlines():
		'''line = line.strip()
		if whetherEmotion(line, threshold):
			emotion_word_set.add(line)'''
		line = line.strip().split()
                for each in line:
                        if whetherEmotion(each, threshold):
                                emotion_word_set.add(each)
	f.close()

	for each in emotion_word_set:
		out.write(each)
		out.write("\n")
	out.close()
