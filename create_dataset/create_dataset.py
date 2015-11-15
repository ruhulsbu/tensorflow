import sys
import getopt
import subprocess

text_path = "./complete_dataset.txt"
tags_path = "./complete_dataset.tags"
suffix = "_pos"
train_count = 60.0
dev_count = 20.0
test_count = 20.0

#python create_dataset.py -x ./complete_dataset.txt -g ./complete_dataset.tags -l 60 -v 20 -t 20 -s _pos

try:
	opts, args = getopt.getopt(sys.argv[1:],"hx:g:l:v:t:s:", ["text=","tags=", "learn=", "validate=", "test=", "suffix="])
except getopt.GetoptError:
	print 'python create_dataset.py -x <te\'x\'t> -g <ta\'g\'s> -l <%\'l\'earn> -v <%\'v\'alidate> -t <%\'t\'est> -s <suffix>'
	print 'example: python create_dataset.py -x complete_dataset.txt -g complete_datset.tags -l 60 -v 20 -t 20 -s _pos'
	sys.exit(2)

for opt, arg in opts:
	if opt == '-h':
		print 'python create_dataset.py -x <te\'x\'t> -g <ta\'g\'s> -l <%\'l\'earn> -v <%\'v\'alidate> -t <%\'t\'est>'
		print 'example: python create_dataset.py -x complete_dataset.txt -g complete_datset.tags -l 60 -v 20 -t 20 -s _pos'
		sys.exit()
	elif opt in ("-x", "--text"):
		text_path = arg
	elif opt in ("-g", "--tags"):
		tags_path = arg
	elif opt in ("-l", "--learn"):
		train_count = float(arg)
	elif opt in ("-v", "--validate"):
		dev_count = float(arg)
	elif opt in ("-t", "--test"):
		test_count = float(arg)
	elif opt in ("-s", "--suffix"):
		suffix = arg
	else:
		sys.exit()

print "Text File: ", text_path
print "Tags File: ", tags_path
print "Train Count(%) = ", train_count
print "Validate Count(%) = ", dev_count
print "Test Count(%) = ", test_count
print "File Suffix = ", suffix


text_reader = open(text_path, 'r')
tags_reader = open(tags_path, 'r')

text_train_writer = open("./train" + suffix + ".txt", 'w')
tags_train_writer = open("./train" + suffix + ".tags", 'w')

text_dev_writer = open("./dev" + suffix + ".txt", 'w')
tags_dev_writer = open("./dev" + suffix + ".tags", 'w')

text_test_writer = open("./test" + suffix + ".txt", 'w')
tags_test_writer = open("./test" + suffix + ".tags", 'w')


output = subprocess.check_output("wc -l " + text_path, shell=True).strip().split(" ")
total_count = float(output[0].strip())
print "Total Lines: ", total_count

count = 0
for sentence in text_reader:
	taginfo = tags_reader.readline()

	if(100.0 * count / total_count < train_count):
		text_train_writer.write(sentence)
		tags_train_writer.write(taginfo)
	elif(100.0 * count / total_count < (train_count + dev_count)):
		text_dev_writer.write(sentence)
		tags_dev_writer.write(taginfo)
	else:
		text_test_writer.write(sentence)
		tags_test_writer.write(taginfo)
	#break
	count += 1

print "Done"
