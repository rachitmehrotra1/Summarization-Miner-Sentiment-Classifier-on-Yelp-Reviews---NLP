import nltk
from sentence import Sentence

class Review(object):
	# Tokenizer for converting a review to a list of sentences. 
	SENT_TOKENIZER = nltk.data.load('tokenizers/punkt/english.pickle')

	def __init__(self, review_dict, business=None):
		# Store review-level metadata
		self.review_id = review_dict['review_id'] #string
		self.user_id = review_dict['user_id'] #string
		self.user_name = review_dict['user_name'] #string
		self.stars = int(review_dict['review_stars']) #int
		self.text = review_dict['text']	#string	

		# if passed, store reference to business this review is about
		if business:
			self.business = business

		# Create the list of sentences for this review
		self.sentences = self.sentence_tokenize(self.text)

	def sentence_tokenize(self, review_text):
		print review_text
		print "\n\n\n\n\n"
		return [Sentence(sent, review=self) for sent in  Review.SENT_TOKENIZER.tokenize(review_text)]

	def __iter__(self):
		return self.sentences.__iter__()

	def __str__(self):
		return self.text
		