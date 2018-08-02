import numpy as np 
import pandas as pd 
import os
import matplotlib 
matplotlib.use('pdf')
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score 
from sklearn.metrics import recall_score
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import LabelEncoder

from scipy.interpolate import spline

import operator
from scipy.stats import wilcoxon

from utils.constants import UNIVARIATE_DATASET_NAMES as DATASET_NAMES
from utils.constants import UNIVARIATE_ARCHIVE_NAMES as ARCHIVE_NAMES
from utils.constants import MAX_PROTOTYPES_PER_CLASS

def zNormalize(x):
    x_mean = x.mean(axis=0) # mean for each dimension of time series x
    x_std = x.std(axis = 0) # std for each dimension of time series x
    x = (x - x_mean)/(x_std)
    return x

def readucr(filename):
    data = np.loadtxt(filename, delimiter = ',')
    Y = data[:,0]
    X = data[:,1:]
    return X, Y

def check_if_file_exits(file_name):
    return os.path.exists(file_name)

def create_directory(directory_path): 
    if os.path.exists(directory_path):
        return None
    else: 
        try: 
            os.makedirs(directory_path)
        except:
            # in case another machine created the path meanwhile
            return None
        return directory_path

def transform_labels(y_train,y_test):
    """
    Transform label to min equal zero and continuous 
    For example if we have [1,3,4] --->  [0,1,2]
    """
    # init the encoder
    encoder = LabelEncoder()
    # concat train and test to fit 
    y_train_test = np.concatenate((y_train,y_test),axis =0)
    # fit the encoder 
    encoder.fit(y_train_test)
    # transform to min zero and continuous labels 
    new_y_train_test = encoder.transform(y_train_test)
    # resplit the train and test
    new_y_train = new_y_train_test[0:len(y_train)]
    new_y_test = new_y_train_test[len(y_train):]
    return new_y_train, new_y_test    

def read_all_datasets(root_dir,archive_name, sort_dataset_name = False):
    datasets_dict = {}

    dataset_names_to_sort = []
    
    for dataset_name in DATASET_NAMES: 
        file_name = root_dir+archive_name+'/'+dataset_name+'/'+dataset_name
        x_train, y_train = readucr(file_name+'_TRAIN')
        x_test, y_test = readucr(file_name+'_TEST')
        datasets_dict[dataset_name] = (x_train.copy(),y_train.copy(),x_test.copy(),y_test.copy())
        dataset_names_to_sort.append((dataset_name,len(x_train)))
    
    item_getter = 1
    if sort_dataset_name == True: 
        item_getter = 0
    dataset_names_to_sort.sort(key=operator.itemgetter(item_getter))
    
    for i in range(len(DATASET_NAMES)):
        DATASET_NAMES[i] = dataset_names_to_sort[i][0]
    
    return datasets_dict

def calculate_metrics(y_true, y_pred,duration,clustering=False):
    """
    Return a data frame that contains the precision, accuracy, recall and the duration
    For clustering it applys the adjusted rand index
    """
    if clustering == False:
        res = pd.DataFrame(data = np.zeros((1,5),dtype=np.float), index=[0], 
            columns=['precision','accuracy','error','recall','duration'])
        res['precision'] = precision_score(y_true,y_pred,average='macro')
        res['accuracy'] = accuracy_score(y_true,y_pred)
        res['recall'] = recall_score(y_true,y_pred,average='macro')
        res['duration'] = duration
        res['error'] = 1-res['accuracy']
        return res
    else: 
        res = pd.DataFrame(data = np.zeros((1,2),dtype=np.float), index=[0], 
            columns=['ari','duration'])
        res['duration']=duration
        res['ari'] = adjusted_rand_score(y_pred,y_true)
        return res

def dataset_is_ready_to_plot(df_res,dataset_name,archive_name,array_algorithm_names):
    for algorithm_name in array_algorithm_names:
                # if any algorithm algorithm is not finished do not plot 
                if not any(df_res.loc[(df_res['dataset_name']==dataset_name) \
                            & (df_res['archive_name']==archive_name)] \
                            ['algorithm_name']==algorithm_name)\
                                       or (df_res.loc[(df_res['dataset_name']==dataset_name) \
                            & (df_res['archive_name']==archive_name)\
                            & (df_res['algorithm_name']==algorithm_name)]\
                                       ['nb_prototypes'].max()!=MAX_PROTOTYPES_PER_CLASS):
                    return False
    return True

def init_empty_df_metrics():
    return pd.DataFrame(data = np.zeros((0,5),dtype=np.float), index=[], 
        columns=['precision','accuracy','error','recall','duration'])

def get_df_metrics_from_avg(avg_df_metrics):
    res = pd.DataFrame(data = np.zeros((1,5),dtype=np.float), index=[0], 
        columns=['precision','accuracy','error','recall','duration'])
    res['accuracy'] = avg_df_metrics['accuracy'].mean()
    res['precision'] = avg_df_metrics['precision'].mean()
    res['error'] = avg_df_metrics['error'].mean()
    res['recall'] = avg_df_metrics['recall'].mean()
    res['duration'] = avg_df_metrics['duration'].mean()
    return res

def get_df_metrics_from_avg_data_cluster(avg_df_metrics):
    res = pd.DataFrame(data = np.zeros((1,2),dtype=np.float), index=[0],
        columns=['ari','duration'])
    res['ari'] = avg_df_metrics['ari'].mean()
    res['duration'] = avg_df_metrics['duration'].mean()
    return res

def read_dataset(root_dir,archive_name,dataset_name):
    datasets_dict = {}

    file_name = root_dir+'/'+archive_name+'/'+dataset_name+'/'+dataset_name
    x_train, y_train = readucr(file_name+'_TRAIN')
    x_test, y_test = readucr(file_name+'_TEST')
    datasets_dict[dataset_name] = (x_train.copy(),y_train.copy(),x_test.copy(),
        y_test.copy())

    return datasets_dict

def plot_epochs_metric(hist, file_name, metric='loss'):
    plt.figure()
    plt.plot(hist.history[metric])
    plt.plot(hist.history['val_'+metric])
    plt.title('model '+metric)
    plt.ylabel(metric)
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(file_name)
    plt.close()

def save_logs(output_directory, hist, y_pred, y_true,duration ):
    hist_df = pd.DataFrame(hist.history)
    hist_df.to_csv(output_directory+'history.csv', index=False)

    df_metrics = calculate_metrics(y_true,y_pred, duration)
    df_metrics.to_csv(output_directory+'df_metrics.csv', index=False)

    index_best_model = hist_df['loss'].idxmin() 
    row_best_model = hist_df.loc[index_best_model]

    df_best_model = pd.DataFrame(data = np.zeros((1,6),dtype=np.float) , index = [0], 
        columns=['best_model_train_loss', 'best_model_val_loss', 'best_model_train_acc', 
        'best_model_val_acc', 'best_model_learning_rate','best_model_nb_epoch'])
    
    df_best_model['best_model_train_loss'] = row_best_model['loss']
    df_best_model['best_model_val_loss'] = row_best_model['val_loss']
    df_best_model['best_model_train_acc'] = row_best_model['acc']
    df_best_model['best_model_val_acc'] = row_best_model['val_acc']
    df_best_model['best_model_learning_rate'] = row_best_model['lr']
    df_best_model['best_model_nb_epoch'] = index_best_model

    df_best_model.to_csv(output_directory+'df_best_model.csv', index=False)

    # for FCN there is no hyperparameters fine tuning - everything is static in code 

    # plot losses 
    plot_epochs_metric(hist, output_directory+'epochs_loss.png')

# visualizations pairwise plots for AALTD 2018

def generate_results_csv(output_file_name, root_dir,root_dir_dataset_archive, add_bake_off=True):
    res = pd.DataFrame(data=np.zeros((0, 7), dtype=np.float), index=[],
                       columns=['classifier_name', 'archive_name', 'dataset_name',
                                'precision', 'accuracy', 'recall', 'duration'])
    CLASSIFIERS = ['resnet','resnet_augment','ensemble']
    ITERATIONS = 1
    for classifier_name in CLASSIFIERS:
        for archive_name in ARCHIVE_NAMES:
            datasets_dict = read_all_datasets(root_dir_dataset_archive, archive_name)
            for it in range(ITERATIONS):
                curr_archive_name = archive_name
                if it != 0:
                    curr_archive_name = curr_archive_name + '_itr_' + str(it)
                for dataset_name in datasets_dict.keys():
                    output_dir = root_dir + '/results/' + classifier_name + '/' \
                                 + curr_archive_name + '/' + dataset_name + '/' + 'df_metrics.csv'
                    if not os.path.exists(output_dir):
                        continue
                    df_metrics = pd.read_csv(output_dir)
                    df_metrics['classifier_name'] = classifier_name
                    df_metrics['archive_name'] = archive_name
                    df_metrics['dataset_name'] = dataset_name
                    res = pd.concat((res, df_metrics), axis=0, sort=False)

    res.to_csv(root_dir + output_file_name, index=False)
    # aggreagte the accuracy for iterations on same dataset
    res = pd.DataFrame({
        'accuracy': res.groupby(
            ['classifier_name', 'archive_name', 'dataset_name'])['accuracy'].mean()
    }).reset_index()

    return res

def plot_pairwise(root_dir,root_dir_dataset_archive, classifier_name_1, classifier_name_2,
                  res_df=None, title='', fig=None, color='green', label=None):
    if fig is None:
        plt.figure()
    else:
        plt.figure(fig)

    if res_df is None:
        res_df = generate_results_csv('results.csv', root_dir,root_dir_dataset_archive)

    sorted_df = res_df.loc[(res_df['classifier_name'] == classifier_name_1) | \
                           (res_df['classifier_name'] == classifier_name_2)]. \
        sort_values(['classifier_name', 'archive_name', 'dataset_name'])
    # number of classifier we are comparing is 2 since pairwise
    m = 2
    # get max nb of ready datasets
    # count the number of tested datasets per classifier
    df_counts = pd.DataFrame({'count': sorted_df.groupby(
        ['classifier_name']).size()}).reset_index()
    # get the maximum number of tested datasets
    max_nb_datasets = df_counts['count'].max()
    min_nb_datasets = df_counts['count'].min()
    # both classifiers should have finished
    assert (max_nb_datasets == min_nb_datasets)

    data = np.array(sorted_df['accuracy']).reshape(m, max_nb_datasets).transpose()

    # concat the dataset name and the archive name to put them in the columns s
    sorted_df['archive_dataset_name'] = sorted_df['archive_name'] + '__' + \
                                        sorted_df['dataset_name']
    # create the data frame containg the accuracies
    df_data = pd.DataFrame(data=data, columns=np.sort([classifier_name_1, classifier_name_2]),
                           index=np.unique(sorted_df['archive_dataset_name']))

    # # assertion
    # p1 = float(sorted_df.loc[(sorted_df['classifier_name'] == classifier_name_1) &
    #                          (sorted_df['dataset_name'] == 'Beef')]['accuracy'])
    # p2 = float(df_data[classifier_name_1]['UCR_TS_Archive_2015__Beef'])
    # assert (p1 == p2)

    x = np.arange(start=0, stop=1, step=0.01)
    plt.xlim(xmax=1.02, xmin=0.0)
    plt.ylim(ymax=1.02, ymin=0.0)

    plt.scatter(x=df_data[classifier_name_1], y=df_data[classifier_name_2], color='blue')
    # c=sorted_df['theme_colors'])
    plt.xlabel('without data augmentation', fontsize='large')
    plt.ylabel('with data augmentation', fontsize='large')
    plt.plot(x, x, color='black')
    # plt.legend(loc='upper left')
    plt.title(title)

    uniq, counts = np.unique(df_data[classifier_name_1] < df_data[classifier_name_2], return_counts=True)
    print('Wins', counts[-1])

    uniq, counts = np.unique(df_data[classifier_name_1] == df_data[classifier_name_2], return_counts=True)
    print('Draws', counts[-1])

    uniq, counts = np.unique(df_data[classifier_name_1] > df_data[classifier_name_2], return_counts=True)
    print('Losses', counts[-1])

    p_value = wilcoxon(df_data[classifier_name_1], df_data[classifier_name_2], zero_method='pratt')[1]
    print(p_value)

    plt.savefig(root_dir + '/' + classifier_name_1 + '-' + classifier_name_2 + '_' + title + '.pdf'
                , bbox_inches='tight')