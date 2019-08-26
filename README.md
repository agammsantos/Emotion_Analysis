# Indonesian-Twitter-Emotion-Dataset
Dataset repository: [Github](https://github.com/meisaputri21/Indonesian-Twitter-Emotion-Dataset)

This dataset contains 4.403 Indonesian tweets which are labeled into five emotion classes: love, anger, sadness, joy and fear. 

## Data Format
Each line consists of a tweet and its respective emotion label separated by semicolon (,). The first line is a header. For a tweet with coma (,) inside the text, there is an quote (" ") to avoid column separation.  </br>
The tweets in this dataset has been pre-processed using the following criterias:
1. Username mention (@) has been replaced with term *[USERNAME]*
2. URL/hyperlink (http://... or https://...) has been replaced with term *[URL]*
3. Sensitive number, such as phone number, invoice number and courier tracking number has been replaced with term *[SENSITIVE-NO]*  

## Important File Information
There are two python program files in the repository, they are:
1. twitter.py: consist of the code for data preparation, data analysis and visualization, and model building.
2. twitterflask.py: consist of the code for the prediction I/O interface, run this file to predict a tweet's emotion!

## Methods Being Compared and Used for Emotion Prediction
The data is separated into two parts. 80% for train and 20% for test. With the help of Count Vectorizer, the methods being used are: 
1. Multinomial Naive Bayes
2. Complement Naive Bayes

The interface for prediction input and output is made with Flask.

## Citation
If you want to publish a paper using this dataset and pre-trained word embedding, please cite this publication: <br />
**Mei Silviana Saputri, Rahmad Mahendra, and Mirna Adriani, "*Emotion Classification on Indonesian Twitter Dataset*", in Proceeding of International Conference on Asian Language Processing 2018. 2018.**


## License
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.
