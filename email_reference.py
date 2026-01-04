import re # regular expression - used to remove certain patterns (HTML, signatures, etc from the data)
import string # used to get the punctuation characters/ lists
from nltk.corpus import stopwords # remove unwanted common words 
from nltk.stem import WordNetLemmatizer #convert the words to the base form 

stop_words= set (stopwords.words("english")) # lists of common words like "the", "is" , "and"
lemmatizer= WordNetLemmatizer() # used to simplify the words(fixing-> fix)

# 1. cleaning the email text

def clean_email(text):
    # 1. Lowercase 
    text= text.lower() # lowers the text so that it doesn't give any confusion 
    #2. remove the email IDs
    text= re.sub(r'\S@\S+', "", text) #removes the irrelvant and nosy data 
    #3. remove the URLS
    text = re.sub(r"http\S+|www\S+", "", text) #removes the irrelvant and noisy url that adds no meaning to the classsification
    #4. remove the punctation 
    text= text.translate(str.maketrans("", "", string.punctuation))# reduces the noise like symbols- !, ? 
    #5. Tokenize
    words= text.split() #i can able to split tthe text into the "words"
    #6. remove the stopwords + lemmatization 
    processed_words= []
    for w in words: 
        if w not in stop_words:
            processed_words.append(lemmatizer.lemmatize(w)) 
            # purpose- removes the useless words, lemmatizes each words- running= run, issues= issue
    return " ".join(processed_words) # returns the cleaned sentence back as a text 
    
sample_email= "Hello team, My internet is not working. please fix this ASAP!!!!"
print(clean_email(sample_email))

#o/p: hello team internet work please fix asap 


# 2. A sample dataset of emails
# they must understand that email classification depends on the data + labels
# Helps them to see how the ML model learns from the data + labels
# required for ML model training

emails = [
    "My internet is not working, I need help immediately",
    "I want to know the status of my refund request",
    "Great service! I appreciate the quick support",
    "You guys keep sending too many mails. Stop it."
]

labels = [
    "complaint",
    "request",
    "feedback",
    "spam"
]

# cleaning the emails using the clean_email function

cleaned_emails= [clean_email(e) for e in emails]
print(cleaned_emails)

#o/p : 
# [
#  'internet work need help immediately',
#  'want know status refund request',
#  'great service appreciate quick support',
#  'guys keep sending many mail stop'
# ]

# Convert the text into numerical vectors using TF-IDF vectorization
# important step and cruical 
# ML models cannot understand the text directly, so that we are converting them to numbers 

## TF-IDF vectorization - learn 

from sklearn.feature_extraction.text import TfidfVectorizer 

vectorizer= TfidfVectorizer()

x = vectorizer.fit_transform(cleaned_emails)

# how the text becomes a numeric matrix 
# why TF-IDF is necessary- other ? alterative? 
# what is feature extractions means 

# 4. train/test spilt 

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    x, labels, test_size=0.25, random_state=42
)

# 5. Subject + Body extraction from email

subjects = [
    "Internet Issue",
    "Refund Request",
    "Appreciation",
    "Unwanted Emails"
]

full_emails = [subjects[i] + " " + emails[i] for i in range(len(emails))]
cleaned_emails = [clean_email(e) for e in full_emails]

# 6. Saving the structured cleaned data

import pandas as pd

df = pd.DataFrame({
    "subject": subjects,
    "body": emails,
    "cleaned_text": cleaned_emails,
    "label": labels
})

print(df)
df.to_csv("cleaned_email_dataset.csv", index=False)

# Optional: Add Urgency Labels

urgency = ["high", "medium", "low", "low"]
df["urgency"] = urgency

#-------------------------------------------------------#
#Milestone 2- 
# step 1: train a baseline classifer - Logistic regression, naive bayes
# step 2: evaluate the model - accuracy, precision, recall, f1-score
# step 3- upgrade into Transformer model- BERT/ DistilBERT


# train logistic regression
from sklearn.linear_model import LogisticRegression # this is a classification algo commonly used for text data 

model = LogisticRegression(max_iter= 1000) 

#model-> our classifer, max_iter -> ensures the model converges (important for the text data)

model.fit(X_train, y_train)

# this is where the learning happens: model sees that TF-IDF vectors + learns patterns linking words-> labels

y_pred= model.predict(X_test)

# model predicts the labels for unseen emails, these predictions will be compared with these true labels 

#step 2: evaluate the model:

from sklearn.metrics import accuracy_score, classification_report

accuracy= accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# accuracy- correctly predicted emails/ total emails 
# simple but not always enough 

report= classification_report(y_test, y_pred)
print("classification report: {report}")

# precision, recall, f1 score -> give a detailed view of model performance
# precision-> correct predictions 
# recall-> actual cases were found
# f1 score-> balance of precision + recall

# this is very mandatory for milestone 2 completion

# introduce the second baseline: Naive Bayes classifier 

## Move into the transformer models like BERT/ DistilBERT for better performance


## packages installed : !pip install transformers torch

import torch 
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

label_map= {
    "compliant": 0,
    "request": 1,
    "spam": 2,
    "feedback": 3
}

numeric_labels = [label_map[1] for 1 in labels]

## Load the tokenizer + model

tokenizer = DistilBertTokenizerFast.from_pretrained(
    "distilbert-base-uncased"
)## loads teh pre-trained model

## tokenise the cleaned emails

encodings= tokenizer(
    texts, ## list of emails 
    truncation= True, # cuts the long emails 
    padding= True, # equal length for batching 
    max_length= 128, # efficient for emails 
)

## create some dataset classes

class EmailDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    
dataset = EmailDataset(encodings, numeric_labels)


## load DistilBert model 

model= DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=4 # 4 classes 
)

# training + evaluation 

from transformers import Trainer, TrainingArguments

training_args= TrainingArguments(
    output_dir="./results",
    num_train_epochs= 3, ## fine-tuning (not heavy - training from scratch)
    per_device_train_batch_size= 8, # safe for the limited GPU
    logging_dir= "./logs", 
    logging_steps= 10,
    save_strategy= "no" # avoids the saving checkpoints for the demo
)

trainer= Trainer(
    model= model,
    args= training_args,
    train_dataset= dataset
)

trainer.train()

## this is the fine-tuning
## the model learns the email-speech patterns + links them to the labels
# loss decreases + accuracy improves over epochs


# testing/ prediction example

test_email = "Internet not working, please fix urgently"
inputs = tokenizer(test_email, return_tensors="pt")

outputs = model(**inputs)
predicted_class = torch.argmax(outputs.logits, dim=1)

reverse_label_map = {v: k for k, v in label_map.items()}
print(reverse_label_map[predicted_class.item()])


# Milestone 3 (Weeks 5–6): Urgency Detection & Scoring
# ●	Objective: Implement urgency prediction.
# ●	Tasks:
# ○	Train urgency classification model.
# ○	Combine ML + keyword-based detection.
# Validate results with confusion matrix & F1 score.

# urgency labels aligned with emails
urgency_labels = [
    "high",     # internet not working immediately
    "medium",   # refund request
    "low",      # appreciation
    "low"       # spam / unsubscribe
]

urgency_map = {
    "low": 0,
    "medium": 1,
    "high": 2
}

y_urgency = [urgency_map[u] for u in urgency_labels]

#step2: rule based urgency detection

high_urgency_keywords = [
    "urgent", "asap", "immediately", "not working", "down", "failed"
]

medium_urgency_keywords = [
    "soon", "please", "request", "help", "issue"
]

# rule-based urgency detection function

def rule_based_urgency(text):
    text = text.lower()
    
    for word in high_urgency_keywords:
        if word in text:
            return "high"

    for word in medium_urgency_keywords:
        if word in text:
            return "medium"
    
    return "low"

print(rule_based_urgency("Internet not working please fix asap"))
print(rule_based_urgency("Need help with refund"))
print(rule_based_urgency("Thank you for the support"))

## high, medium, low

# combine ML + rule-based approach

## reuse the TF-IDF features because- same text -> diff tasks -> diff labels, features stays the same, the targets change

X_urgency= vectorizer.fit_transform(cleaned_emails)

# train/test spilt

X_train_u, X_test_u, y_train_u, y_test_u = train_test_split(
    X_urgency, y_urgency, test_size=0.25, random_state=42)


# train logistic regression for urgency

urgency_model =LogisticRegression(max_iter= 1000)
urgency_model.fit(X_train_u, y_train_u)

# pred urgency 

y_pred_u= urgency_model.predict(X_test_u)


## hybrid - urgency scoring (rule + ML)

reverse_urgency_map = {0: "low", 1: "medium", 2: "high"}

def hybrid_urgency_detection(text):
    # step 1: rule-based check 
    
    rule_result= rule_based_urgency(text)
    
    if rule_result == "high":
        return "high"
    
    # step2: ML prediction
    
    cleaned= clean_email(text)
    vec= vectorizer.transform ([cleaned])
    ml_pred = urgency_model.predict(vec)[0]
    
    return reverse_urgency_map[ml_pred]


print(hybrid_urgency_detection("System is down since morning"))
print(hybrid_urgency_detection("Please help with refund status"))
print(hybrid_urgency_detection("Thanks for the update"))


# step 5: evaluation using confusion matrix + F1 score 

from sklearn.metrics import confusion_matrix, f1_score

print(confusion_matrix(y_test_u, y_pred_u))
f1= f1_score(y_test_u, y_pred_u, average="weighted")
print(f"F1 Score: {f1}")