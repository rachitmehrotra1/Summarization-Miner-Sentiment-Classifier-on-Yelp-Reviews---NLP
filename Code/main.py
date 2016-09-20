import json
import time
import pandas as pd
import sys
#Rachit Mehrotra
#rm4149@nyu.edu

from pymongo import MongoClient
from classes.business import Business

def get_reviews_for_business(bus_id, df):
	return df[df.business_id==bus_id]

def read_data():
	return pd.read_csv('/Users/Rachit/Downloads/NLP Project/opinion-mining-master/processed.csv')

def main(): 
	reload(sys)
	sys.setdefaultencoding('ISO-8859-1')
	print sys.getdefaultencoding()
	client = MongoClient()
	print "client data"
	print client
	db = client.yelptest2
	print "db data"
	print db
	summaries_coll = db.summaries	
	print "summmaries_coll"
	print summaries_coll

	print "Loading data..."
	df = read_data()
	bus_ids = df.business_id.unique()[21:]

	for bus_id in bus_ids:

		print "Working on biz_id %s" % bus_id
		start = time.time()

		biz = Business(get_reviews_for_business(bus_id,df))
		print "biz data"
		print biz
		print "reached summarizarion"
		summary = biz.aspect_based_summary()
		print "Inserting into summaries col"
		print summary
		summaries_coll.insert(summary)

		print "Inserted summary for %s into Mongo" % biz.business_name

		elapsed = time.time() - start
		print "Time elapsed: %d" % elapsed


if __name__ == "__main__":
	main()
