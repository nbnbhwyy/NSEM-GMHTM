from itertools import count
import os
from typing import Counter
import numpy as np
import pickle
from numpy.core.fromnumeric import shape
from scipy.sparse import csr_matrix
import pandas as pd
# from collections import Counter
import gensim
from gensim.parsing.preprocessing import STOPWORDS
from collections import defaultdict

def save_pkl(path, obj):
	with open(path, 'wb') as f:
		pickle.dump(obj, f)
		print(" [*] save {}" .format(path))

def load_pkl(path):
	with open(path,'rb') as f:
		obj = pickle.load(f)
		print(" [*] load {}" .format(path))
		return obj

class TextReader():
	def __init__(self, data_path):
		print("Dataset preparing....")
		self.base_path = data_path
		self.train_path = os.path.join(self.base_path, "raw/train.txt")
		self.valid_path = os.path.join(self.base_path, "raw/valid.txt")
		self.test_path = os.path.join(self.base_path, "raw/test.txt")
		self.vocab_path = os.path.join(self.base_path, "processed/vocab.pkl")


		try: 
			self._load()
		except:
			self._rebuild_vocab()

		self.idx2word = {v:k for k, v in self.vocab.items()}
		self.vocab_size = len(self.vocab)
		print(f"Load {self.vocab_size} words..")
	

	def _read_text(self, file_path):
		labels = []
		texts = []
		with open(file_path, encoding='utf-8') as f:
			docs = open(file_path).read().split('\n')
			for n_i, line in enumerate(docs):
				if line == '':
					continue
				label, text = line.strip().split('\t',1)
				text = list(gensim.utils.tokenize(text, lower=True)) # 分词
				texts.append(text)
				labels.append(label)

			return labels, texts

	def _read_data(self, train_path, valid_path, test_path):
		train_label, self.train_text = self._read_text(train_path)
		valid_label, self.valid_text = self._read_text(valid_path)
		test_label, self.test_text = self._read_text(test_path)
		
	def _build_vocab(self, frequence=50):

		self.label_dict = {}
		with open(self.train_path.replace('train.txt', 'labels.txt'),'r') as f:
			f = f.read()
			l = f.split('\n')
			for id, label in enumerate(l):
				self.label_dict[label.lower()] = id

		self._read_data(self.train_path, self.valid_path, self.test_path)

		all_text = self.train_text + self.valid_text + self.test_text

		# build dictionary
		dictionary = gensim.corpora.Dictionary(all_text)
		print("Before",len(dictionary))
		dictionary = self._filter_(dictionary, frequence)
		print("After",len(dictionary))

		self.vocab = dictionary

		save_pkl(self.vocab_path, self.vocab)
	
	def _filter_(self, dictionary, frequence=50):
		dictionary.filter_tokens(list(map(dictionary.token2id.get, STOPWORDS)))

		len_1_words = list(filter(lambda w: len(w) == 1, dictionary.values()))
		dictionary.filter_tokens(list(map(dictionary.token2id.get, len_1_words)))

		dictionary.filter_extremes(no_below=frequence)
		dictionary.compactify()

		return dictionary

	def _rebuild_vocab(self, frequence=50):
		self._build_vocab(frequence)
		self.train_data, self.train_label = self._file_to_data(self.train_path)
		self.valid_data, self.valid_label = self._file_to_data(self.valid_path)
		self.test_data, self.test_label = self._file_to_data(self.test_path)


	def _file_to_data(self, file_path):
		labels, texts = self._read_text(file_path)

		data = []
		m_labels = []
		for label, text in zip(labels, texts):

			if len(self.vocab.doc2bow(text)) < 3:
				continue

			word = list(map(self.vocab.token2id.get, text))
			word = np.array(list(filter(lambda x: x is not None, word)))
			
			if label.isdigit():
				m_labels.append(int(label))
			else:
				m_labels.append(self.label_dict[label.lower()])
			data.append(word)
		
		lens = list(map(len, data))
		print(" [*] load {} docs, avg len: {}, max len: {}".format(len(data), np.mean(lens), np.max(lens)))
		

		data = np.array(data)
		save_pkl(file_path.replace('/raw/','/processed/').replace('.txt','_data.pkl'), data)

		m_labels = np.array(m_labels)
		if m_labels.max() == self.get_n_classes():
			m_labels = m_labels - 1
		save_pkl(file_path.replace('/raw/','/processed/').replace('.txt','_label.pkl'), m_labels)
		return data, m_labels

	def _load(self):
		# 加载语料
		self._read_data(self.train_path, self.valid_path, self.test_path)

		processed_train_path = self.train_path.replace('/raw/','/processed/').replace('.txt','_data.pkl')
		processed_valid_path = self.valid_path.replace('/raw/','/processed/').replace('.txt','_data.pkl')
		processed_test_path = self.test_path.replace('/raw/','/processed/').replace('.txt','_data.pkl')

		self.vocab = load_pkl(self.vocab_path)

		self.train_data = load_pkl(processed_train_path)
		self.train_label = load_pkl(processed_train_path.replace('_data.pkl', '_label.pkl')).astype(np.int)

		self.valid_data = load_pkl(processed_valid_path)
		self.valid_label = load_pkl(processed_valid_path.replace('_data.pkl', '_label.pkl')).astype(np.int)

		self.test_data = load_pkl(processed_test_path)
		self.test_label = load_pkl(processed_test_path.replace('_data.pkl', '_label.pkl')).astype(np.int)
	
	def get_n_classes(self):
		if hasattr(self, "n_classes"):
			return self.n_classes
		else:
			with open(self.base_path + '/raw/labels.txt','r') as f:
				f = f.read()
				l = f.split('\n')
				self.n_classes = len(l)
				return self.n_classes

	def get_sequence(self, data_type):
		if data_type == "train":
			return (self.train_data, self.train_label, self.train_text)

		elif data_type == "valid":
			return (self.valid_data, self.valid_label, self.valid_text)

		elif data_type == "test":
			return (self.test_data, self.test_label, self.test_text)

		elif data_type == "train+valid":
			data = np.concatenate([self.train_data, self.valid_data])
			label = np.concatenate([self.train_label, self.valid_label])
			text = np.concatenate([self.train_text, self.valid_text])
			return(data, label, text)

		elif data_type == "all":
			data = np.concatenate([self.train_data, self.valid_data, self.test_data])
			label = np.concatenate([self.train_label, self.valid_label, self.test_label])
			text = np.concatenate([self.train_text, self.valid_text, self.test_text])
			return(data, label, text)
		else:
			raise Exception(" [!] Unkown data type : {}" .format(data_type))

	def generator_sequence(self, data_type = "train", batch_size=32, rand=True):
		raw_data, raw_label, raw_text = self.get_sequence(data_type)
		
		count = 0
		while True:
			if not rand:
				beg = (count * batch_size) % raw_data.shape[0]
				end = ((count + 1) * batch_size) % raw_data.shape[0]
				if beg > end:
					beg -= raw_data.shape[0]

				idx = np.arange(beg,end)
			else:
				idx = np.random.randint(0, len(raw_data), batch_size)
				
			data = raw_data[idx]
			label = raw_label[idx]
			text = raw_text[idx]

			yield [data, label, text]

			count += 1
	

	def get_matrix(self, data_type = "train", mode='onehot', normalize=False):
		raw_data, raw_label, raw_text = self.get_sequence(data_type)

		x = np.zeros((len(raw_data), self.vocab_size))
		for i, seq in enumerate(raw_data):
			counter = defaultdict(int)

			for j in seq:
				counter[j] += 1
			
			for j,c in list(counter.items()):
				if mode == 'count':
					x[i][j] = c
				elif mode == 'fred':
					x[i][j] = c / len(seq)
				elif mode == 'onehot':
					x[i][j] = 1
				elif mode == 'tfidf':
					tf = 1 + np.log(c)
					idf = np.log(1 + self.vocab.num_docs /(1 + self.vocab.dfs[j]))

					x[i][j] = tf * idf
				else:
					raise ValueError('Unknown vectorization mode:', mode)
		
		if normalize:
			x = x / x.sum(axis=1, keepdims=True)
			x = x.astype(np.float32)

		return [x, raw_label, raw_text]
	

	def generator_matrix(self, data_type = "train", batch_size=32, rand=True, mode='onehot', normalize=False):
		raw_data, raw_label, raw_text = self.get_sequence(data_type)
		
		count = 0
		while True:
			if not rand:
				beg = (count * batch_size) % raw_data.shape[0]
				end = ((count + 1) * batch_size) % raw_data.shape[0]
				if beg > end:
					beg -= raw_data.shape[0]

				idx = np.arange(beg,end	)
			else:
				idx = np.random.randint(0, len(raw_data), batch_size)
			
			data = raw_data[idx]
			label = raw_label[idx]
			text = raw_text[idx]

			x = np.zeros((len(data), self.vocab_size))
			for i, seq in enumerate(data):
				counter = defaultdict(int)

				for j in seq:
					counter[j] += 1

				for j,c in list(counter.items()):
					if mode == 'count':
						x[i][j] = c
					elif mode == 'fred':
						x[i][j] = c / len(seq)
					elif mode == 'onehot':
						x[i][j] = 1
					elif mode == 'tfidf':
						tf = 1 + np.log(c)
						idf = np.log(1 + self.vocab.num_docs /(1 + self.vocab.dfs[j]))

						x[i][j] = tf * idf
					else:
						raise ValueError('Unknown vectorization mode:', mode)
			
			if normalize:
				x = x / x.sum(axis=1, keepdims=True)
				x = x.astype(np.float32)
				
			yield [x, label, text]

			count += 1

	def get_bow(self, data_type = "train"):
		raw_data, raw_label, raw_text = self.get_sequence(data_type)

		bow = []
		for item in raw_data:
			bow.append(list(Counter(item).items()))
		return bow
	
	def get_sparse_matrix(self, data_type = "train", mode='onehot', normalize=False):
		raw_data, raw_label, raw_text = self.get_sequence(data_type)

		
		row = []
		col = []
		data = []
		for i, seq in enumerate(raw_data):
			counter = defaultdict(int)
			# 统计每个id出现的次数
			for j in seq:
				counter[j] += 1
			# 填充矩阵元素
			for j,c in list(counter.items()):
				row.append(i)
				col.append(j)
				if mode == 'count':
					data.append(c)
				elif mode == 'fred':
					data.append( c / len(seq))
				elif mode == 'onehot':
					data.append( 1)
				elif mode == 'tfidf':
					tf = 1 + np.log(c)
					idf = np.log(1 + self.vocab.num_docs /(1 + self.vocab.dfs[j]))
					data.append( tf * idf )
				else:
					raise ValueError('Unknown vectorization mode:', mode)
		
		data, row, col = np.array(data), np.array(row), np.array(col)

		x = csr_matrix((data, (row, col)), shape=(len(raw_data), self.vocab_size)).astype(np.float32)
		return [x, raw_label, raw_text]

	def get_sparse_matrix_single_cell(self, data_path = "", data_name = ""):
		csv = pd.read_csv(data_path+data_name,sep=',',index_col=0)
		#label = pd.read_csv(data_path+'CellBenchCEL-2 seq5cl10x5cl_label.csv',sep=',',index_col=0)
		return  csr_matrix(csv).astype(np.float32)#,label
