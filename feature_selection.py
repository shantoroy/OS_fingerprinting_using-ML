# Modified from source: https://machinelearningmastery.com/feature-selection-machine-learning-python/

# Feature Selection with Univariate Statistical Tests
from pandas import read_csv
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2


# select best features from all features using ANOVA (f_classif())
def univariate_stat(df, names, no_of_best):
    print("##############################")
    print("########## f_classif #########")
    print("##############################")

    # considering the last column as class labels
    array = df.values
    X = array[:,0:len(names)-1]
    Y = array[:,len(names)-1]

    stat_list = [f_classif, chi2]

    for stat_test in stat_list:
    # feature extraction
        test = SelectKBest(score_func=stat_test, k=no_of_best)
        fit = test.fit(X, Y)

        # summarize scores
        set_printoptions(precision=3)
        # print(fit.scores_)
        
        score = {}

        for i,j in zip(names, list(fit.scores_)):
            score[i] = j

        feature_scores = dict(sorted(score.items(), key=lambda item: item[1], reverse=True))
        # print(feature_scores)
        print("")
        print("{:<15} {:<10}".format('Feature','Score'))
        for k, v in feature_scores.items():
            print("{:<15} {:<10}".format(k, v))
    
    
    
# Feature Extraction with RFE
from pandas import read_csv
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


def recursive_feature_eliminate(df, names, no_of_best):
    print("##############################")
    print("############# RFE ############")
    print("##############################")
    

    # considering the last column as class labels
    array = df.values
    X = array[:,0:len(names)-1]
    Y = array[:,len(names)-1]
    
    # feature extraction
    model = LogisticRegression(solver='lbfgs')
    rfe = RFE(model, no_of_best)
    fit = rfe.fit(X, Y)
    
    # print("Num Features: %d" % fit.n_features_)
    # print("Selected Features: %s" % fit.support_)
    # print("Feature Ranking: %s" % fit.ranking_)
    
    selection = {}
    for i,j in zip(names, list(fit.ranking_)):
        selection[i] = j
        
    support = {}
    for i,j in zip(names, list(fit.support_)):
        support[i] = j
    # print(support)
    print("{:<15} {:<10}".format('Feature','Support'))
    for k, v in support.items():
        print("{:<15} {:<10}".format(k, v))

    feature_rank = dict(sorted(selection.items(), key=lambda item: item[1]))
    # print(feature_rank)
    print("")
    print("{:<15} {:<10}".format('Feature','Rank'))
    for k, v in feature_rank.items():
        print("{:<15} {:<10}".format(k, v))
    
    
    
    
    
# Feature Importance with Extra Trees Classifier
from pandas import read_csv
from sklearn.ensemble import ExtraTreesClassifier

def extra_tree_classifier(df, names):
    print("##############################")
    print("#### ExtraTreesClassifier ####")
    print("##############################")

    # considering the last column as class labels
    array = df.values
    X = array[:,0:len(names)-1]
    Y = array[:,len(names)-1]
    
    # feature extraction
    model = ExtraTreesClassifier(n_estimators=10)
    model.fit(X, Y)
    # print(model.feature_importances_)
    
    importance = {}

    for i,j in zip(names, list(model.feature_importances_)):
        importance[i] = j

    feature_importance = dict(sorted(importance.items(), key=lambda item: item[1], reverse=True))
    # print(feature_importance)
    print("{:<15} {:<10}".format('Feature','Importance'))
    for k, v in feature_importance.items():
        print("{:<15} {:<10}".format(k, v))
    
    
    
    
if __name__ == "__main__":
    filename = "labeled_dataset.csv"
    # filename = input("Enter the filename: ")
    names = ['ip.hdr_len', 'ip.flags.rb',\
         'ip.flags.df', 'ip.flags.mf', 'ip.frag_offset', 'ip.ttl',\
         'ip.len', 'tcp.seq', 'tcp.ack', 'tcp.len', \
         'tcp.hdr_len', 'tcp.flags.fin', 'tcp.flags.syn', 'tcp.flags.reset',\
         'tcp.flags.push', 'tcp.flags.ack', 'tcp.flags.urg', 'tcp.flags.cwr', 'tcp.window_size',\
         'tcp.urgent_pointer', 'os']
    df = read_csv(filename, usecols=names)
    # no_of_best = int(input("Enter the no. of best features: "))
    no_of_best = 10
    
    print("")
    univariate_stat(df, names, no_of_best)
    print("")
    recursive_feature_eliminate(df, names, no_of_best)
    print("")
    extra_tree_classifier(df, names)
