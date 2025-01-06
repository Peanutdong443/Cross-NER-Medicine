
import os
import subprocess
import pickle
import logging
import time
import random
from datetime import timedelta
from sklearn.manifold import TSNE
from tqdm import tqdm
from transformers import DataCollatorForTokenClassification
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression

from src.dataloader import domain2labels


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

def map_matrix_to_entitylabels(X, label_list, entitylabel_list, is_numpy=False):
    '''
        map matrix X to entitylabels
        including combine B-/I- and change the sequence 
        (columns for each labels)
    '''
    if not is_numpy:
        X_device = X.device
        new_X = torch.zeros((X.shape[0],len(entitylabel_list)))
        new_X = new_X.to(X_device)
    else:
        new_X = np.zeros((X.shape[0],len(entitylabel_list)))

    # BIO or BIOES
    is_BIOES = False
    for tmp in list(label_list):
        if tmp.find('E-')!=-1 or tmp.find('S-')!=-1:
            is_BIOES = True
            break

    for idx, item in enumerate(entitylabel_list):
        if item in ['others','I-misc','B-misc']:
            new_X[:,idx] = X[:,list(label_list).index('O')]
        else:
            if is_BIOES:
                new_X[:,idx] = X[:,list(label_list).index('B-'+item)] + \
                            X[:,list(label_list).index('I-'+item)] + \
                            X[:,list(label_list).index('E-'+item)] + \
                            X[:,list(label_list).index('S-'+item)]
            else:
                new_X[:,idx] = X[:,list(label_list).index('B-'+item)] + \
                            X[:,list(label_list).index('I-'+item)]
    return new_X

def map_vec_to_entitylabels(x, label_list, entitylabel_list, is_numpy=False):
    '''
        map vector x (y_list) to entitylabels
        including combine B-/I- and change the sequence 
        (columns for each labels)
    '''
    if not is_numpy:
        x_device = x.device
        new_x = x.clone().detach().cpu()
        new_x = new_x.to(x_device)
    else:
        new_x = np.array(x)

    ignore_idx = [-100]
    other_idx = [list(label_list).index(l) for l in ['O']]
    for idx, y in enumerate(x):
        if not is_numpy:
            _y = y.item()

        if _y in other_idx:
            new_x[idx] = list(entitylabel_list).index('others')
        elif _y in ignore_idx:
            continue
        else:
            item = label_list[_y]
            new_x[idx] = list(entitylabel_list).index(item[2:])
    
    return new_x

def add_temperature(preds, temperature):
    return preds/temperature

def get_entity_mask(label_list):
    mask = [False if l in ['O'] else True for l in label_list]
    return mask

def get_label_center_graph(X, Y, label_list, is_plot=False):
    '''
        build the graph according to the center of each label
    '''

    # filter out masked label
    X, Y = filter_masked_label(X=X,
                                Y=Y, 
                                masked_label=[-100])
    # exclude the other and miscellous label
    class_select_mask = get_entity_mask(label_list)
    X, Y = filter_selected_label(X=X,
                                Y=Y,
                                label_list=label_list,
                                class_select_mask=class_select_mask)
    # get the center
    class_center, class_seen_mask = get_center(X, Y, num_class=len(label_list))
    


    return class_center, class_seen_mask


def get_feature_center_graph(X, Y, label_list, is_plot=False):
    '''
        build the graph according to the center of each label
    '''

    # filter out masked label
    X, Y = filter_masked_label(X=X,
                               Y=Y,
                               masked_label=[-100])
    # exclude the other and miscellous label
    class_select_mask = get_entity_mask(label_list)
    X, Y = filter_selected_label(X=X,
                                 Y=Y,
                                 label_list=label_list,
                                 class_select_mask=class_select_mask)
    # get the center
    class_center, class_seen_mask = get_center1(X, Y, num_class=None)

    if is_plot:
        # t-SNE visualization
        index_list = list(range(X.shape[0]))
        random.shuffle(index_list)
        X = X[:2000, :]
        Y = Y[:2000]
        plot_distribution(X=X, Y=Y,
                          label_list=label_list,
                          class_select_mask=class_seen_mask,
                          class_center=class_center)

    return class_center, class_seen_mask
def get_center(X, Y, num_class=None):
    """
    input:
        X : (N*D) tensor. There are N samples and each sample is a D-dimensional vector
        Y : N-dimensional tensor (label index from 0 ~ D-1)
    return:
        class_center: (C*D) tensor, each class center is a D-dimensional vector,
                        and C is the number of the seen class
        class_seen_mask: a list represents the seen classes
    """
    # ensure X and Y in the same divice
    X_device = X.device
    Y = Y.to(X_device)

    # set the number of classes
    if num_class==None:
        num_class = int(X.shape[1])

    # get the mask for the class whose center can be calculated
    class_seen_mask = [True if i in Y else False for i in range(num_class)] 
    class_unseen_mask = [True if i not in Y else False for i in range(num_class)]
    num_class_unseen = int(np.sum(class_unseen_mask)) 

    # add dummy samples for the unseen class
    unseen_class_index = torch.where(torch.tensor(class_unseen_mask))[0].to(X_device)
    Y = torch.cat((Y, unseen_class_index))
    unseen_class_X = torch.zeros((num_class_unseen,X.shape[1])).to(X_device)
    X = torch.cat((X, unseen_class_X), dim=0)

    # convert to one-hot label
    Y = torch.eye(num_class)[Y.long()].to(X_device)

    # get center for all classes
    U1 = torch.diag(1 / torch.sum(Y, dim=0))
    T1 = torch.matmul(U1, Y.T)
    class_center = torch.matmul(T1, X)
    class_center = class_center[class_seen_mask, :]

    #class_center = torch.matmul(torch.matmul(torch.diag(1/torch.sum(Y, dim=0)),Y.T),X)
    class_center = class_center[class_seen_mask,:]

    return class_center, class_seen_mask
def get_center1(X, Y, num_class=None):
    """
    input:
        X : (N*D) tensor. There are N samples and each sample is a D-dimensional vector
        Y : N-dimensional tensor (label index from 0 ~ D-1)
    return:
        class_center: (C*D) tensor, each class center is a D-dimensional vector,
                        and C is the number of the seen class
        class_seen_mask: a list represents the seen classes
    """
    # ensure X and Y in the same divice
    X_device = X.device
    Y = Y.to(X_device)

    # set the number of classes
    if num_class==None:
        num_class = int(X.shape[0])

    # get the mask for the class whose center can be calculated
    class_seen_mask = [True if i in Y else False for i in range(num_class)]
    # class_unseen_mask = [True if i not in Y else False for i in range(num_class)]
    # num_class_unseen = int(np.sum(class_unseen_mask))
    #
    # # add dummy samples for the unseen class
    # unseen_class_index = torch.where(torch.tensor(class_unseen_mask))[0].to(X_device)
    # Y = torch.cat((Y, unseen_class_index))
    # unseen_class_X = torch.zeros((num_class_unseen,X.shape[1])).to(X_device)
    # X = torch.cat((X, unseen_class_X), dim=0)

    # convert to one-hot label
    # Y = torch.eye(num_class)[Y.long()].to(X_device)

    # get center for all classes
    # u=torch.diag(1/torch.sum(Y, dim=0))
    # t=torch.matmul(u,Y.T)
    # class_center = torch.matmul(t,X)
    # class_center = class_center[class_seen_mask,:]
    class_center=X
    return class_center, class_seen_mask
def plot_embedding(X, Y, label_list, label_indexes, has_center=False):
    '''
        plot the score embedding in 2-D space
    '''
    # scale to 0-1
    x_min, x_max = torch.min(X, 0)[0], torch.max(X, 0)[0]
    X = (X - x_min) / (x_max - x_min)

    num_tag = len(label_list)
    if has_center:
        num_sample = X.shape[0] - num_tag
        class_center = X[-num_tag:,:]
        X = X[:-num_tag,:] 
    else:
        num_sample = X.shape[0]

    color_lst = torch.zeros(num_sample)

    for i, l in enumerate(label_indexes):
        class_mask = torch.eq(Y, l)
        color_lst[class_mask] = i
        print(class_mask.sum())

    plt.scatter(X[:,0], 
                X[:,1], 
                c=color_lst, 
                marker='.')
    if has_center:
        plt.scatter(class_center[:,0],
                    class_center[:,1], 
                    c=list(range(num_tag)), 
                    marker='*')
        for i in range(num_tag):
            plt.text(class_center[i][0],
                    class_center[i][1],
                    s=str(label_list[i]),
                    size=15)
    
def filter_masked_label(X, Y, masked_label=[], is_numpy=False):
    '''
        filter out the masked labels
    '''
    for l in masked_label:
        if is_numpy:
            mask = np.not_equal(Y, l)
        else:
            mask = torch.not_equal(Y, l)
        Y = Y[mask]
        X = X[mask]
    return X, Y

def filter_selected_label(X, Y, label_list, class_select_mask):
    '''
        select the samples of the given classes
    '''
    for i, l in enumerate(label_list):
        if class_select_mask[i] == False:
            mask = torch.not_equal(Y, i)
            Y = Y[mask]
            X = X[mask]
    return X, Y

def plot_distribution(X, Y, label_list, class_select_mask, class_center=None):
    '''
        plot the score/embedding distribution of all the class in the target domain
    '''
    _X = X.clone().detach().cpu()
    _Y = Y.clone().detach().cpu()
    _class_center = class_center.clone().detach().cpu()
    # plot the graph of distribution for each class
    select_label_list = label_list[class_select_mask]
    label_indexes = torch.where(torch.tensor(class_select_mask))[0]

    # t-SNE for visualization
    tsne = TSNE(n_components=2)
    if _class_center!= None:
        concat_X = torch.cat((_X, _class_center),dim=0)
        low_dim_representation = torch.tensor(tsne.fit_transform(concat_X))
        plot_embedding(low_dim_representation, _Y, select_label_list, label_indexes, has_center=True)
        plt.show()
    else:
        low_dim_representation = torch.tensor(tsne.fit_transform(_X))
        plot_embedding(low_dim_representation, _Y, select_label_list, label_indexes, has_center=False)
        plt.show()
    
def init_experiment(params, logger_filename):
    """
    Initialize the experiment:
    - save parameters
    - create a logger
    """
    # save parameters
    get_saved_path(params)
    pickle.dump(params, open(os.path.join(params.dump_path, "params.pkl"), "wb"))

    # create a logger
    logger = create_logger(os.path.join(params.dump_path, logger_filename))
    logger.info('============ Initialized logger ============')
    logger.info('\n'.join('%s: %s' % (k, str(v))
                          for k, v in sorted(dict(vars(params)).items())))
    logger.info('The experiment will be stored in %s\n' % params.dump_path)

    return logger


class LogFormatter():

    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (
            record.levelname,
            time.strftime('%x %X'),
            timedelta(seconds=elapsed_seconds)
        )
        message = record.getMessage()
        message = message.replace('\n', '\n' + ' ' * (len(prefix) + 3))
        return "%s - %s" % (prefix, message) if message else ''


def create_logger(filepath):
    # create log formatter
    log_formatter = LogFormatter()
    
    # create file handler and set level to debug
    if filepath is not None:
        file_handler = logging.FileHandler(filepath, "a")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(log_formatter)

    # create console handler and set level to info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    # create logger and set level to debug
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if filepath is not None:
        logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # reset logger elapsed time
    def reset_time():
        log_formatter.start_time = time.time()
    logger.reset_time = reset_time

    return logger


def get_saved_path(params):
    """
    create a directory to store the experiment
    """
    dump_path = "./" if params.dump_path == "" else params.dump_path
    if not os.path.isdir(dump_path):
        subprocess.Popen("mkdir -p %s" % dump_path, shell=True).wait()
    assert os.path.isdir(dump_path)

    # create experiment path if it does not exist
    exp_path = os.path.join(dump_path, params.exp_name)
    if not os.path.exists(exp_path):
        subprocess.Popen("mkdir -p %s" % exp_path, shell=True).wait()
    
    # generate id for this experiment
    if params.exp_id == "":
        chars = "0123456789"
        while True:
            exp_id = "".join(random.choice(chars) for _ in range(0, 3))
            if not os.path.isdir(os.path.join(exp_path, exp_id)):
                break
    else:
        exp_id = params.exp_id
    # update dump_path
    params.dump_path = os.path.join(exp_path, exp_id)
    if not os.path.isdir(params.dump_path):
        subprocess.Popen("mkdir -p %s" % params.dump_path, shell=True).wait()
    assert os.path.isdir(params.dump_path)

def set_random_seed():
    seed = int(random.random()*1000000%1000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_label_prompt(name):
     if name=="ai":
         ori_label_token_map= {"I-person": ['Michael', 'John', 'David', 'Mark', 'Martin', 'Paul'],
         "I-organisation": ['Inc', 'Co', 'Corp', 'Party', 'Association', '&'],
         "I-location": ['Germany', 'Australia', 'England', 'France', 'Italy', 'Belgium'],
         "I-misc": ['German', 'Cup', 'French', 'Israeli', 'Australian', 'Olympic'],
         "I-field": ['learning', 'imaging', 'analysis', 'language', 'recognition', 'Design'],
         "I-task": ['tagging', 'parsing', 'synthesis', 'extraction', 'reduction', "detection"],
         "I-product": ['Graphs', 'checkers', 'processor', 'robot', 'control', 'industrial'],
         "I-algorithm": ['descriptors', 'principal', 'linear', 'matching', 'algorithm', 'Markov'],
         "I-researcher": ['Scheinman', 'Felix', 'Schmidhuber', 'Fred', 'David', 'Boris'],
         "I-metrics": ['noise', 'maximum', 'Understudy', 'Hinge', 'loss', 'function'],
         "I-programlang": ['C', '+', 'Octave', 'Java', 'programming', 'python'],
         "I-conference": ['Computationa', 'Linguistics', 'Computer', 'Institute', 'Technology', 'Group'],
         "I-university": ['University', 'Mellon', 'Saarbruecken', 'California', 'Toronto', 'Centrale'],
         "I-country": ['Africa', 'South', 'Kingdom', 'United', 'States', 'Union']
         }
         return ori_label_token_map
     elif name=="PG":
        ori_label_token_map = {
                               "I-Gene_or_gene_product": ['americium', 'europium', 'curium', 'gadolinium', 'copper', 'lead'],
                               "I-Complex": ['carbon', 'Glyceraldehyde', 'Cyclic', 'carbon', 'hydrogen', 'oxygen'],
                               "I-Cellular_component": ['Saturn', 'Titan', 'Enceladus', 'Ganymede', 'Jupiter', 'Achilles'],
                               "I-Simple_chemical": ['Chemical', 'Accounts', 'Angewandte', 'Astrophysical', 'Nature', 'Journal'],
                               }
        return ori_label_token_map
     elif name=="CG":
        ori_label_token_map = {"I-Tissue": ['Michael', 'John', 'David', 'Mark', 'Martin', 'Paul'],
                               "I-Anatomical_system": ['Inc', 'Co', 'Corp', 'Party', 'Association', '&'],
                               "I-Amino_acid": ['Germany', 'Australia', 'England', 'France', 'Italy', 'Belgium'],
                               "I-Cancer": ['German', 'Cup', 'French', 'Israeli', 'Australian', 'Olympic'],
                               "I-Cell": ['August', 'Wolf', 'Edgar', 'Nüsslein-Volhard', 'Peter', 'Richard'],
                               "I-Organ":  ['United', 'Russia', 'Japan', 'UK', 'England', 'Italy'],
                               "I-Multi-tissue_structure": ['clinical', 'enzymology', 'chemistry', 'medicine', 'genetics', 'CRISPR'],
                               "I-Developing_anatomical_structure": ['Acetylcholinesterase', 'DNA', 'RNA', 'Spermidine', 'Alkaline', 'T7'],
                               "I-Organism_subdivision": ['RNase', 'Histone', 'nuclear', 'calponin', 'Piwi', 'Argonaute'],
                               "I-Simple_chemical": ['americium', 'europium', 'curium', 'gadolinium', 'copper', 'lead'],
                               "I-Cellular_component": ['carbon', 'Glyceraldehyde', 'Cyclic', 'carbon', 'hydrogen', 'oxygen'],
                               "I-Immaterial_anatomical_entity": ['season', 'Gobaith', 'Summer', 'Championships', 'communism', 'World'],
                               "I-Organism_substance": ['Saturn', 'Titan', 'Enceladus', 'Ganymede', 'Jupiter', 'Achilles'],
                               "I-Gene_or_gene_product": ['Chemical', 'Accounts', 'Angewandte', 'Astrophysical', 'Nature', 'Journal'],
                               "I-Pathological_formation": ['stem', 'Nicolaus', 'metaphysics', 'theory', 'Protein-DNA', 'dichroism'],
                               "I-Organism": ['Zealand', 'States', 'Kingdom', 'United', 'New', 'The']
                               }
        return ori_label_token_map
     elif name=="music":
        ori_label_token_map = {"I-person": ['Michael', 'John', 'David', 'Mark', 'Martin', 'Paul'],
                               "I-organisation": ['Inc', 'Co', 'Corp', 'Party', 'Association', '&'],
                               "I-location": ['Germany', 'Australia', 'England', 'France', 'Italy', 'Belgium'],
                               "I-misc": ['German', 'Cup', 'French', 'Israeli', 'Australian', 'Olympic'],
                               "I-musicgenre": ['Rock', 'Punk', 'music', 'hip', 'hop', 'Latin'],
                               "I-song": ['Hole', 'Freelove', 'Is', 'a', 'You', "Be"],
                               "I-band": ['Specials', 'Town', 'Spice', 'Earth', 'Band', 'Leyland'],
                               "I-album": ['Come', 'Barbra', 'Idiot', 'Nirvana', 'Norz', 'Please'],
                               "I-musicalinstrument": ['accordion', 'piano', 'six-string', 'guitar', 'banjo', '&'],
                               "I-award": ['Music', 'Awards', 'lobe', 'Movie', 'Outstanding', 'Rock'],
                               "I-musicalartist": ['Brennan', 'Pearlman', 'Boine', 'Murray', 'Thomas', 'Ronstadt'],
                               "I-event": ['season', 'Gobaith', 'Summer', 'Championships', 'communism', 'World'],
                               "I-country": ['Zealand', 'States', 'Kingdom', 'United', 'New', 'The']
                               }
        return ori_label_token_map

     elif name == "science":
        ori_label_token_map = {"I-person": ['Michael', 'John', 'David', 'Mark', 'Martin', 'Paul'],
                               "I-organisation": ['Inc', 'Co', 'Corp', 'Party', 'Association', '&'],
                               "I-location": ['Germany', 'Australia', 'England', 'France', 'Italy', 'Belgium'],
                               "I-misc": ['German', 'Cup', 'French', 'Israeli', 'Australian', 'Olympic'],
                               "I-scientist": ['August', 'Wolf', 'Edgar', 'Nüsslein-Volhard', 'Peter', 'Richard'],
                               "I-university": ['Oxford', 'University', 'Oak', 'Uppsala', 'Uppsala', "Heidelberg"],
                               "I-country":  ['United', 'Russia', 'Japan', 'UK', 'England', 'Italy'],
                               "I-discipline": ['clinical', 'enzymology', 'chemistry', 'medicine', 'genetics', 'CRISPR'],
                               "I-enzyme": ['Acetylcholinesterase', 'DNA', 'RNA', 'Spermidine', 'Alkaline', 'T7'],
                               "I-protein": ['RNase', 'Histone', 'nuclear', 'calponin', 'Piwi', 'Argonaute'],
                               "I-chemicalelement": ['americium', 'europium', 'curium', 'gadolinium', 'copper', 'lead'],
                               "I-chemicalcompound": ['carbon', 'Glyceraldehyde', 'Cyclic', 'carbon', 'hydrogen', 'oxygen'],
                               "I-event": ['season', 'Gobaith', 'Summer', 'Championships', 'communism', 'World'],
                               "I-astronomicalobject": ['Saturn', 'Titan', 'Enceladus', 'Ganymede', 'Jupiter', 'Achilles'],
                               "I-academicjournal": ['Chemical', 'Accounts', 'Angewandte', 'Astrophysical', 'Nature', 'Journal'],
                               "I-theory": ['stem', 'Nicolaus', 'metaphysics', 'theory', 'Protein-DNA', 'dichroism'],
                               "I-award": ['Zealand', 'States', 'Kingdom', 'United', 'New', 'The']
                               }
        return ori_label_token_map
     elif name == "politics":
        ori_label_token_map = {"I-person": ['Michael', 'John', 'David', 'Mark', 'Martin', 'Paul'],
                               "I-organisation": ['Inc', 'Co', 'Corp', 'Party', 'Association', '&'],
                               "I-location": ['Germany', 'Australia', 'England', 'France', 'Italy', 'Belgium'],
                               "I-misc": ['German', 'Cup', 'French', 'Israeli', 'Australian', 'Olympic'],
                               "I-politician": ['Nenad', 'Nigel', 'Joseph', 'Tony', 'Zappa', 'Fritz'],
                               "I-election": ['2011', 'state', '1920', '1997', '2002', "2007"],
                               "I-politicalparty": ['Italian', 'Texas', 'Gibraltar', 'Panhellenic', 'Party', 'LIDER'],
                               "I-event": ['season', 'Gobaith', 'Summer', 'Championships', 'communism', 'World'],
                               "I-country": ['Africa', 'South', 'Kingdom', 'United', 'States', 'Union']
                               }
        return ori_label_token_map
     elif name=="literature":
        ori_label_token_map = {"I-person": ['Michael', 'John', 'David', 'Mark', 'Martin', 'Paul'],
                               "I-organisation": ['Inc', 'Co', 'Corp', 'Party', 'Association', '&'],
                               "I-location": ['Germany', 'Australia', 'England', 'France', 'Italy', 'Belgium'],
                               "I-misc": ['German', 'Cup', 'French', 'Israeli', 'Australian', 'Olympic'],
                               "I-book": ['Rabbit', 'Fountainhead', 'Atlas', 'Prince', 'Archia', 'Pantoum'],
                               "I-writer": ['David', 'Caitriona', 'Erdrich', 'Adrian', 'Tzara', "Ion"],
                               "I-award": ['Music', 'Awards', 'lobe', 'Movie', 'Outstanding', 'Rock'],
                               "I-event": ['season', 'Gobaith', 'Summer', 'Championships', 'communism', 'World'],
                               "I-country": ['Africa', 'South', 'Kingdom', 'United', 'States', 'Union'],
                               "I-poem": ['Ikke', 'Nuances', 'The', 'Chanson', 'Mother', 'Freedom'],
                               "I-magazine": ['Yorker', 'Atlantic', 'Geographic', 'Life', 'TV', 'Yorker'],
                               "I-literarygenre": ['poems', 'novels', 'travel', 'science', 'fiction', "stories"],
                               }
        return ori_label_token_map
class DataCollatorForLMTokanClassification(DataCollatorForTokenClassification):
    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None
        ori_labels = [feature['ori_labels'] for feature in features] if 'ori_labels' in features[0].keys() else None
        # poss = [feature['poss'] for feature in features] if 'poss' in features[0].keys() else None
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="pt" if labels is None else None,
        )

        if labels is None:
            return batch

        sequence_length = torch.tensor(batch["input_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            batch["labels"] = [label + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels]
            batch['ori_labels'] = [label + [self.label_pad_token_id] * (sequence_length - len(label)) for label in
                                   ori_labels]
            # batch['poss'] = [pos + [100] * (sequence_length - len(pos)) for pos in
            #                        poss]
        else:
            batch["labels"] = [[self.label_pad_token_id] * (sequence_length - len(label)) + label for label in labels]
            batch["ori_labels"] = [[self.label_pad_token_id] * (sequence_length - len(label)) + label for label in
                                   ori_labels]
            # batch['poss'] = [[100] * (sequence_length - len(pos)) + pos for pos in
            #                        poss]
        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
        return batch


def add_label_token_bert(model, tokenizer, label_map):
    sorted_add_tokens = sorted(list(label_map.keys()), key=lambda x: len(x), reverse=True)
    tokenizer.add_tokens(sorted_add_tokens)
    num_tokens, _ = model.bert.embeddings.word_embeddings.weight.shape
    model.resize_token_embeddings(len(sorted_add_tokens) + num_tokens)
    for token in sorted_add_tokens:
        if token.startswith('B-') or token.startswith('I-'):  # 特殊字符
            index = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token))
            if len(index) > 1:
                raise RuntimeError(f"{token} wrong split: {index}")
            else:
                index = index[0]
            # assert index>=num_tokens, (index, num_tokens, token)
            if isinstance(label_map[token], list):
                indexes = tokenizer.convert_tokens_to_ids(label_map[token])
            else:
                indexes = tokenizer.convert_tokens_to_ids([label_map[token]])

            embed = model.bert.embeddings.word_embeddings.weight.data[indexes[0]]

            # Calculate mean vector if there are multiple label words.
            for i in indexes[1:]:
                embed += model.bert.embeddings.word_embeddings.weight.data[i]
            embed /= len(indexes)
            model.bert.embeddings.word_embeddings.weight.data[index] = embed

    return tokenizer