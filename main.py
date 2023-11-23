import requests
import csv
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

url_ml_one = 'http://ieeexploreapi.ieee.org/api/v1/search/articles?apikey=zkn22d5qwu42d44tkmmn4sch&format=json&max_records=200&start_record=1&sort_order=asc&sort_field=author&abstract=Machine+learning'
url_ml_two = 'http://ieeexploreapi.ieee.org/api/v1/search/articles?apikey=zkn22d5qwu42d44tkmmn4sch&format=json&max_records=200&start_record=1&sort_order=asc&sort_field=article_title&abstract=Machine+learning'
url_ml_three = ' http://ieeexploreapi.ieee.org/api/v1/search/articles?apikey=zkn22d5qwu42d44tkmmn4sch&format=json&max_records=200&start_record=1&sort_order=asc&sort_field=publication_title&abstract=Machine+learning'
url_ml_four = 'http://ieeexploreapi.ieee.org/api/v1/search/articles?apikey=zkn22d5qwu42d44tkmmn4sch&format=json&max_records=200&start_record=1&sort_order=desc&sort_field=article_number&abstract=Machine+learning'

url_img_one = 'http://ieeexploreapi.ieee.org/api/v1/search/articles?apikey=zkn22d5qwu42d44tkmmn4sch&format=json&max_records=200&start_record=1&sort_order=asc&sort_field=author&abstract=Computer+Vision'
url_img_two = 'http://ieeexploreapi.ieee.org/api/v1/search/articles?apikey=zkn22d5qwu42d44tkmmn4sch&format=json&max_records=200&start_record=1&sort_order=asc&sort_field=article_number&abstract=Computer+Vision'
url_img_three = 'http://ieeexploreapi.ieee.org/api/v1/search/articles?apikey=zkn22d5qwu42d44tkmmn4sch&format=json&max_records=200&start_record=1&sort_order=asc&sort_field=article_title&abstract=Computer+Vision'
url_img_four = 'http://ieeexploreapi.ieee.org/api/v1/search/articles?apikey=zkn22d5qwu42d44tkmmn4sch&format=json&max_records=200&start_record=1&sort_order=asc&sort_field=publication_title&abstract=Computer+Vision'

urls = {url_ml_one, url_ml_two, url_ml_three, url_ml_four}
count = 1

for url in urls:
    response = requests.get(url)
    #print(response.status_code)
    data = response.json()
    #print(data)

    bad_chars = ['.', '”', '“', 'ˆ', '/', ':', '-', '(', ')', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ';','[', ']', ',', '"','<', '>', '"', '=', '…', 'div', "’", '\'']

    bad_words = ['we', 'the', 'in', 'a', 'during', 'and', 'an', 'this', 'these', 'that', 'paper', 'review', 'more', 'our', 'purpose', 'from','to', 'ass', 'as', 'public', 'live', 'study', 'along', 'purpose', 'open', 'job', 'main', 'h', 'couple','year', 'service','user', 'new', 'present', 'report', 'use', 'within', 'using', 'work', 'task', 'always']

    for i in data['articles']:
        abstract = i['abstract']

        for chars in bad_chars:
            abstract = abstract.replace(chars, '')

        tokens = nltk.word_tokenize(abstract)

        lower_words = [lw_word.lower() for lw_word in tokens]
        remove_bad_words = [word for word in lower_words if not word in bad_words]

        stop_words = set(stopwords.words('english'))
        with_out_stop_words = [w for w in remove_bad_words if not w in stop_words]

        # stem_abs = [stemmer.stem(word) for word in with_out_stop_words]

        final_abs = ''
        for word in with_out_stop_words:
            final_abs = final_abs + lemmatizer.lemmatize(word) + ' '

        id = 'ML_' + str(count)
        count = count + 1

        lable = '1'

        #print(id , final_abs)

        # Machine Learning = 1
        # Big data = 2
        # Data mining = 3
        # Computer vision = 4
        # Bioinformatics = 5
        # Artificial intelligence = 6

        with open('ieee_dataset.csv', mode='a', encoding="utf-8", newline='') as file:

            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([id, final_abs, lable])

            print(id + ' added')