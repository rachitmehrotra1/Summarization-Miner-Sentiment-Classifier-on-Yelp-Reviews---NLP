import nltk
import numpy as np

from nltk.stem.wordnet import WordNetLemmatizer

from transformers.tokenizers import MyPottsTokenizer
from transformers.featurizers import MetaFeaturizer, SubjFeaturizer, LiuFeaturizer
from transformers.asp_extractors import SentenceAspectExtractor

class Sentence(object):

	# Tokenizer for converting a raw string (sentence) to a list of strings (words)
	WORD_TOKENIZER = MyPottsTokenizer(preserve_case=False)
	
	# Lemmatizer
	LEMMATIZER = WordNetLemmatizer()

	# Featurizer
	FEATURIZER = MetaFeaturizer([SubjFeaturizer(), LiuFeaturizer()]) #combine two featurizer objects

	# Aspect Extractor
	ASP_EXTRACTOR = SentenceAspectExtractor()

	def __init__(self, raw, review=None):
		
		self.raw = raw #string
		self.tokenized = self.word_tokenize(raw) #list of strings
		self.pos_tagged = self.pos_tag(self.tokenized) #list of tuples
		self.lemmatized = self.lemmatize(self.pos_tagged) #list of tuples

		if review: #if passed, store a reference to the review this came from
			self.review = review
			self.stars = self.review.stars # star pointer to number of stars (for featurization)

		# compute and store aspects for this sentence
		self.aspects = self.compute_aspects()

	def word_tokenize(self, raw):
		return Sentence.WORD_TOKENIZER.tokenize(raw)

	def pos_tag(self, tokenized_sent):
		return nltk.pos_tag(tokenized_sent)

	def lemmatize(self, pos_tagged_sent):
		lemmatized_sent = []

		# Logic to use POS tag if possible
		for wrd, pos in pos_tagged_sent:
			try: 
				lemmatized_sent.append((Sentence.LEMMATIZER.lemmatize(wrd, pos), pos))
			except KeyError:
				lemmatized_sent.append((Sentence.LEMMATIZER.lemmatize(wrd), pos))

		return lemmatized_sent

	def get_features(self, asarray = False):
		if not hasattr(self, 'features'):
			self.features = Sentence.FEATURIZER.featurize(self)

		if not asarray:
			return self.features

		else:
			return np.array([val for _, val in self.features.iteritems()])

	def compute_aspects(self):
		return Sentence.ASP_EXTRACTOR.get_sent_aspects(self)

	def has_aspect(self, asp_string):
		# re-tokenize the aspect
		asp_toks = asp_string.split(" ")

		# return true if all the aspect tokens are in this sentence 
		return all([tok in self.tokenized for tok in asp_toks])

	def encode(self):
		return {'text': self.raw,
				'user': self.review.user_name
				}

	def __str__(self):
		return self.raw
