import os
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
# import sys


class DataPreprocessing:
    def __init__(self, complaint_docs):
        self.complaint_docs = complaint_docs

    def clean(self):
        print("reading csv file: new_corpus.csv ...")
        nlpCorpus = pd.read_csv(self.complaint_docs, sep='\t')
        #drop nan values
        nlpCorpus = nlpCorpus[nlpCorpus['text'].notna()].reset_index(drop=True)
        print("dropping all but GR and DN motionresultcodes and one hot encoding...")
        GR_DN_nlpCorpus = nlpCorpus[(nlpCorpus["MotionResultCode"] == "GR") | (nlpCorpus["MotionResultCode"] == "DN")]
        #one hot encode labels
        one_hot = pd.get_dummies(nlpCorpus["MotionResultCode"])
        GR_DN_nlpCorpus = GR_DN_nlpCorpus.drop("MotionResultCode", axis=1)
        GR_DN_nlpCorpus = GR_DN_nlpCorpus.join(one_hot)
        GR_DN_nlpCorpus = GR_DN_nlpCorpus.drop("DN", axis=1)
        GR_DN_nlpCorpus.rename(columns={"GR" : "MotionResultCode"}, inplace=True)
        # print(GR_DN_nlpCorpus)
        return GR_DN_nlpCorpus



    def train_test_seperation(self, cleaned_data):
        # stratified training testing split
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        X = cleaned_data
        y = cleaned_data["MotionResultCode"]
        for train_index, test_index in split.split(X, y):
            strat_train_set = cleaned_data.loc[train_index]
            strat_test_set = cleaned_data.loc[test_index]
        print("finishing Training and Testing separation ...")
        print("Length of training set: ", len(strat_train_set))
        print("Length of testing set: ", len(strat_test_set))
        #---------------------------------------------------------------------------
        #write training and testing to directory
        directory = "TrainTest"
        cwd = os.getcwd()
        path = os.path.join(cwd, directory)
        os.mkdir(path)
        print("Directory '% s' created" % directory)
        #---------------------------------------------------------------------------
        strat_train_set.to_csv(os.path.join(path, "TrainingData.csv"), sep="\t", index=False)
        strat_test_set.to_csv(os.path.join(path, "TestingData.csv"), sep="\t", index=False)

        return strat_train_set



    def cross_validation_data_prep(self, strat_train_set):
        #stratified k fold
        kfold = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
        X = strat_train_set
        y = strat_train_set["MotionResultCode"]
        #---------------------------------------------------------------------------
        #make directory for outputed kfold cv set
        directory = "CrossValidationData"
        cwd = os.getcwd()
        path = os.path.join(cwd, directory)
        os.mkdir(path)
        print("Directory '% s' created" % directory)
        #---------------------------------------------------------------------------
        count = 0
        for strat_train_kfold_index, strat_test_kfold_index in kfold.split(X, y):
            X_train_kfold, X_test_kfold = X.iloc[list(strat_train_kfold_index)], X.iloc[list(strat_test_kfold_index)]
            # print(type(X_train_kfold))
            # print(type(X_test_kfold))
            print(X_train_kfold)
            print(X_test_kfold)
            X_train_kfold.to_csv(os.path.join(path, f"CV_TrainData_{count}.csv"), sep="\t", index=False)
            X_test_kfold.to_csv(os.path.join(path, f"CV_TestData_{count}.csv"), sep="\t", index=False)
            count += 1
            # y_train_kfold, y_test_kfold = y.loc[strat_train_kfold_index], y.loc[strat_test_kfold_index]



def main():
    dataProccess = DataPreprocessing("./GR_DN_corpus.csv")
    cleaned = dataProccess.clean()
    train_set = dataProccess.train_test_seperation(cleaned)
    dataProccess.cross_validation_data_prep(train_set)

main()


# 70/30 split on training and testing data
# sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
#
# for train_index, test_index in sss.split(X, y):
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
#
# # 70/30 split on training and validation data
# sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
#
# for train_index, valid_index in sss1.split(X_train, y_train):
#     X_train, X_valid = X[train_index], X[valid_index]
#     y_train, y_valid = y[train_index], y[valid_index]
#
# training_df = pd.DataFrame(X_train).join(y_train)
# validation_df = pd.DataFrame(X_valid).join(y_valid)
# testing_df = pd.DataFrame(X_test).join(y_test)
#
# training_df.to_csv('shariq_training_data.csv', index=False)
# validation_df.to_csv('shariq_valid_data.csv', index=False)
# testing_df.to_csv('shariq_testing_data.csv', index=False)
