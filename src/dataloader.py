
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import DataLoader
import numpy as np

from tqdm import tqdm
import random
import logging
import os
logger = logging.getLogger()

from src.config import get_params
params = get_params()
from transformers import AutoTokenizer
auto_tokenizer = AutoTokenizer.from_pretrained(params.model_name)
pad_token_label_id = nn.CrossEntropyLoss().ignore_index

conll2003_labels =  ['O', "B-person", "I-person", "B-location", "I-location", 
                    'B-organisation', 'I-organisation', 'B-misc', 'I-misc']
politics_labels =   ['O', "B-person", "I-person", "B-location", "I-location", 
                    "B-organisation", "I-organisation", 'B-misc', 'I-misc', 'B-country', 
                    'B-politician', 'I-politician', 'B-election', 'I-election', 'I-country',
                    'B-politicalparty', 'I-politicalparty', 'B-event', 'I-event']
science_labels =    ['O', "B-person", "I-person", "B-location", "I-location", 
                    "B-organisation", "I-organisation", 'B-misc', 'I-misc', 'B-scientist', 
                    'I-scientist', 'B-university', 'I-university', 'B-country', 'I-country',
                    'B-discipline', 'I-discipline', 'B-enzyme', 'I-enzyme', 'B-protein', 
                    'I-protein', 'B-chemicalelement', 'I-chemicalelement', 
                    'B-chemicalcompound', 'I-chemicalcompound', 'B-astronomicalobject', 
                    'I-astronomicalobject', 'B-academicjournal', 'I-academicjournal', 
                    'B-event', 'I-event', 'B-theory', 'I-theory', 'B-award', 'I-award']
music_labels =      ['O', "B-person", "I-person", "B-location", "I-location", 
                    "B-organisation", "I-organisation", 'B-misc', 'I-misc', 
                    'B-musicgenre', 'I-musicgenre', 'B-song', 'I-song', 'B-band',
                    'I-band', 'B-album', 'I-album', 'B-musicalartist', 'I-musicalartist',
                    'B-musicalinstrument', 'I-musicalinstrument', 'B-award', 'I-award', 
                    'B-event', 'I-event', 'B-country', 'I-country']
literature_labels = ["O", "B-person", "I-person", "B-location", "I-location", 
                    "B-organisation", "I-organisation", 'B-misc', 'I-misc', 
                    "B-book", "I-book", "B-writer", "I-writer", "B-award", 
                    "I-award", "B-poem", "I-poem", "B-event", "I-event", 
                    "B-magazine", "I-magazine", "B-literarygenre", "I-literarygenre", 
                    'B-country', 'I-country']
ai_labels =         ["O", "B-person", "I-person", "B-location", "I-location", 
                    "B-organisation", "I-organisation", "B-misc", "I-misc", "B-field", 
                    "I-field", "B-task", "I-task", "B-product", "I-product", "B-algorithm",
                    "I-algorithm", "B-researcher", "I-researcher", "B-metrics", 
                    "I-metrics", "B-programlang", "I-programlang", "B-conference", 
                    "I-conference", "B-university", "I-university", "B-country", 
                    "I-country"]
# movie_labels =      ["O", "B-TITLE", "I-TITLE", "B-GENRE", "I-GENRE", "B-SONG", "I-SONG", "B-YEAR", "I-YEAR", "B-ACTOR", "I-ACTOR", "B-CHARACTER", "I-CHARACTER", "B-RATING", "I-RATING", "B-PLOT", "I-PLOT", "B-REVIEW", "I-REVIEW", "B-DIRECTOR", "I-DIRECTOR", "B-RATINGS_AVERAGE", "I-RATINGS_AVERAGE", "B-TRAILER", "I-TRAILER"]
movie_labels =      ["O","B-Actor","I-Actor","B-Plot", "I-Plot",'B-Opinion','I-Opinion',
                    'B-Award','I-Award','B-Year','I-Year','B-Genre','I-Genre',
                    'B-Director','I-Director', 'B-Soundtrack','I-Soundtrack',
                    'B-Relationship','I-Relationship','B-Character_Name',
                    'I-Character_Name','B-Origin','I-Origin','B-Quote','I-Quote']
restaurant_labels = ["O", "B-Rating", "I-Rating", "B-Location", "I-Location", "B-Amenity",
                    "I-Amenity", "B-Cuisine", "I-Cuisine", "B-Hours", "I-Hours", "B-Price",
                    "I-Price", "B-Dish", "I-Dish", "B-Restaurant_Name", "I-Restaurant_Name"]
atis_labels =   ['O','B-toloc.country_name', 'I-toloc.country_name', 'B-return_time.period_mod', 'I-return_time.period_mod', 'B-depart_time.end_time', 'I-depart_time.end_time', 'B-or', 'I-or', 'B-fromloc.airport_name', 'I-fromloc.airport_name', 'B-class_type', 'I-class_type', 'B-connect', 'I-connect', 'B-arrive_date.month_name', 'I-arrive_date.month_name', 'B-airline_code', 'I-airline_code', 'B-arrive_date.date_relative', 'I-arrive_date.date_relative', 'B-arrive_time.period_mod', 'I-arrive_time.period_mod', 'B-return_date.month_name', 'I-return_date.month_name', 'B-depart_date.day_number', 'I-depart_date.day_number', 'B-day_name', 'I-day_name', 'B-return_date.day_name', 'I-return_date.day_name', 'B-mod', 'I-mod', 'B-meal_code', 'I-meal_code', 'B-state_code', 'I-state_code', 'B-meal_description', 'I-meal_description', 'B-flight_stop', 'I-flight_stop', 'B-depart_time.period_mod', 'I-depart_time.period_mod', 'B-depart_time.time_relative', 'I-depart_time.time_relative', 'B-flight_days', 'I-flight_days', 'B-toloc.airport_name', 'I-toloc.airport_name', 'B-flight_mod', 'I-flight_mod', 'B-fromloc.airport_code', 'I-fromloc.airport_code', 'B-return_date.day_number', 'I-return_date.day_number', 'B-depart_date.day_name', 'I-depart_date.day_name', 'B-restriction_code', 'I-restriction_code', 'B-toloc.airport_code', 'I-toloc.airport_code', 'B-depart_date.year', 'I-depart_date.year', 'B-airport_name', 'I-airport_name', 'B-arrive_time.start_time', 'I-arrive_time.start_time', 'B-time', 'I-time', 'B-flight_number', 'I-flight_number', 'B-day_number', 'I-day_number', 'B-airport_code', 'I-airport_code', 'B-toloc.state_name', 'I-toloc.state_name', 'B-toloc.city_name', 'I-toloc.city_name', 'B-economy', 'I-economy', 'B-meal', 'I-meal', 'B-arrive_time.time', 'I-arrive_time.time', 'B-stoploc.state_code', 'I-stoploc.state_code', 'B-return_time.period_of_day', 'I-return_time.period_of_day', 'B-period_of_day', 'I-period_of_day', 'B-fromloc.city_name', 'I-fromloc.city_name', 'B-round_trip', 'I-round_trip', 'B-cost_relative', 'I-cost_relative', 'B-time_relative', 'I-time_relative', 'B-fare_amount', 'I-fare_amount', 'B-today_relative', 'I-today_relative', 'B-arrive_date.day_name', 'I-arrive_date.day_name', 
                'B-flight_time', 'I-flight_time', 'B-toloc.state_code', 'I-toloc.state_code', 'B-return_date.date_relative', 'I-return_date.date_relative', 'B-stoploc.city_name', 'I-stoploc.city_name', 'B-fare_basis_code', 'I-fare_basis_code', 'B-city_name', 'I-city_name', 

                'B-fromloc.state_code', 'I-fromloc.state_code', 'B-arrive_time.period_of_day', 'I-arrive_time.period_of_day', 'B-depart_time.time', 'I-depart_time.time', 'B-stoploc.airport_name', 'I-stoploc.airport_name', 'B-state_name', 'I-state_name', 'B-month_name', 'I-month_name', 'B-depart_date.today_relative', 'I-depart_date.today_relative', 'B-depart_date.date_relative', 'I-depart_date.date_relative', 'B-depart_time.period_of_day', 'I-depart_time.period_of_day', 'B-aircraft_code', 'I-aircraft_code', 'B-depart_date.month_name', 'I-depart_date.month_name', 'B-return_date.today_relative', 'I-return_date.today_relative', 'B-arrive_date.day_number', 'I-arrive_date.day_number', 'B-arrive_time.end_time', 'I-arrive_time.end_time', 'B-transport_type', 'I-transport_type', 'B-days_code', 'I-days_code', 'B-fromloc.state_name', 'I-fromloc.state_name', 'B-depart_time.start_time', 'I-depart_time.start_time', 'B-arrive_date.today_relative', 'I-arrive_date.today_relative', 'B-airline_name', 'I-airline_name', 'B-arrive_time.time_relative', 'I-arrive_time.time_relative',
                 'B-compartment','I-compartment','B-stoploc.airport_code','I-stoploc.airport_code','B-booking_class','I-booking_class','B-flight','I-flight']
wnut17_labels = ["O","B-location","I-location","B-product","I-product","B-creative-work",
                "I-creative-work","B-group","I-group","B-corporation","I-corporation",
                "B-person", "I-person"]
ontonotes5_labels = ["O","B-ORG","B-PERSON","B-EVENT","B-DATE","B-WORK_OF_ART","B-PERCENT","B-CARDINAL","B-TIME","B-GPE","B-QUANTITY","B-ORDINAL","B-NORP","B-LOC","B-MONEY","B-LAW","B-FAC","B-PRODUCT","B-LANGUAGE",
                        "I-ORG","I-PERSON","I-EVENT","I-DATE","I-WORK_OF_ART","I-PERCENT","I-CARDINAL","I-TIME","I-GPE","I-QUANTITY","I-ORDINAL","I-NORP","I-LOC","I-MONEY","I-LAW","I-FAC","I-PRODUCT","I-LANGUAGE",
                        "E-ORG","E-PERSON","E-EVENT","E-DATE","E-WORK_OF_ART","E-PERCENT","E-CARDINAL","E-TIME","E-GPE","E-QUANTITY","E-ORDINAL","E-NORP","E-LOC","E-MONEY","E-LAW","E-FAC","E-PRODUCT","E-LANGUAGE",
                        "S-ORG","S-PERSON","S-EVENT","S-DATE","S-WORK_OF_ART","S-PERCENT","S-CARDINAL","S-TIME","S-GPE","S-QUANTITY","S-ORDINAL","S-NORP","S-LOC","S-MONEY","S-LAW","S-FAC","S-PRODUCT","S-LANGUAGE"]
PG_labels=[ 'O', 'B-Simple_chemical', 'I-Simple_chemical',
    'B-Gene_or_gene_product','I-Gene_or_gene_product',  'B-Cellular_component', 'I-Cellular_component','B-Complex','I-Complex']
CG_Labels=['I-Tissue', 'I-Anatomical_system', 'B-Multi-tissue_structure', 'B-Organism_subdivision', 'B-Simple_chemical', 'B-Cancer', 'I-Amino_acid', 'I-Cancer', 'I-Cell', 'B-Organism', 'I-Organ', 'B-Cellular_component', 'B-Amino_acid', 'B-Pathological_formation', 'I-Multi-tissue_structure', 'I-Developing_anatomical_structure', 'B-Organism_substance', 'B-Cell', 'I-Organism_subdivision', 'O', 'B-Gene_or_gene_product', 'I-Simple_chemical', 'I-Cellular_component', 'I-Immaterial_anatomical_entity', 'I-Organism_substance', 'B-Tissue', 'I-Gene_or_gene_product', 'B-Immaterial_anatomical_entity', 'B-Organ', 'I-Pathological_formation', 'I-Organism', 'B-Developing_anatomical_structure', 'B-Anatomical_system']
domain2labels = {
    "conll2003": conll2003_labels,
    "politics": politics_labels, 
    "science": science_labels, 
    "music": music_labels, 
    "literature": literature_labels, 
    "ai": ai_labels, 
    "movie": movie_labels, 
    "atis": atis_labels,
    "restaurant": restaurant_labels,
    "wnut17": wnut17_labels,
    "ontonotes5": ontonotes5_labels,
    "PG":PG_labels,
    "CG":CG_Labels
}

domain2entitylabels = {
    "conll2003": ['person','location','organisation','misc','others'],
    "PG":['Simple_chemical','Gene_or_gene_product','Cellular_component','Complex','others'],
    "CG":['Tissue','Anatomical_system','Amino_acid','Cancer','Cell','Organ','Multi-tissue_structure','Developing_anatomical_structure','Organism_subdivision','Simple_chemical','Cellular_component','Immaterial_anatomical_entity','Organism_substance','Gene_or_gene_product','Pathological_formation','Organism','others'],
    "politics":  ['person','location','organisation','misc',
                    'politician','country','election','politicalparty','event','others'],
    "science":   ['person','location','organisation','misc',
                    'scientist','university','country','discipline','enzyme','protein',
                    'chemicalelement','chemicalcompound','astronomicalobject',
                    'academicjournal','event','theory','award','others'],
    "music":    ['person','location','organisation','misc',
                    'musicgenre','song','band','album','musicalartist', 
                    'musicalinstrument','award','event','country','others'],
    "literature":['person','location','organisation','misc',
                    "book","writer","award","poem","event", 
                    "magazine","literarygenre",'country','others'],
    "ai":       ['person','location','organisation','misc',
                    "field","task", "product","algorithm","researcher","metrics",
                    "programlang","conference","university","country",'others'],
    # "movie":    ["TITLE", "GENRE", "SONG","YEAR","ACTOR","CHARACTER",
    #                 "RATING","PLOT","REVIEW","DIRECTOR","RATINGS_AVERAGE","TRAILER",'others'],
    "movie":    ["Actor","Plot", 'Opinion','Award','Year','Genre','Director','Origin',
                    'Soundtrack','Relationship','Character_Name','Quote','others'],
    "restaurant":["Rating","Location","Amenity","Cuisine","Hours","Price","Dish",
                    "Restaurant_Name","others"],
    "atis":     ['return_date.month_name', 'round_trip', 'return_date.today_relative', 'arrive_time.period_of_day',
                 'fromloc.state_code', 'return_date.day_number', 'arrive_date.date_relative', 'depart_date.day_number',
                 'depart_time.period_of_day', 'mod', 'return_date.day_name', 'time_relative', 'flight_stop', 'depart_time.start_time',
                 'time', 'arrive_time.period_mod', 'today_relative', 'toloc.state_code', 'airline_code', 'depart_time.time_relative',
                 'depart_date.day_name', 'arrive_date.day_number', 'days_code', 'airline_name', 'airport_name', 'flight_days', 'economy',
                 'period_of_day', 'connect', 'transport_type', 'city_name', 'meal_code', 'fare_basis_code', 'or', 'stoploc.airport_name',
                 'depart_time.period_mod', 'stoploc.city_name', 'arrive_time.time', 'aircraft_code', 'fromloc.state_name', 'flight_mod',
                 'fromloc.city_name', 'arrive_date.today_relative', 'depart_time.end_time', 'state_name', 'depart_date.today_relative',
                 'toloc.airport_name', 'cost_relative', 'day_name', 'toloc.airport_code', 'stoploc.state_code', 'depart_date.month_name',
                 'meal', 'arrive_date.month_name', 'fromloc.airport_name', 'toloc.state_name', 'return_date.date_relative', 'restriction_code',
                 'depart_date.date_relative', 'flight_time', 'fare_amount', 'month_name', 'airport_code', 'flight_number', 'arrive_date.day_name',
                 'return_time.period_of_day', 'arrive_time.end_time', 'depart_date.year', 'return_time.period_mod', 'arrive_time.start_time', 'fromloc.airport_code',
                 'toloc.city_name', 'arrive_time.time_relative', 'day_number', 'meal_description', 'state_code', 'depart_time.time', 'class_type', 'toloc.country_name','compartment','stoploc.airport_code','booking_class','flight','others'],
    "wnut17":   ["location","product","creative-work","group","corporation","person","others"],
    "ontonotes5":["ORG","PERSON","EVENT","DATE","WORK_OF_ART","PERCENT","CARDINAL","TIME",
                "GPE","QUANTITY","ORDINAL","NORP","LOC","MONEY","LAW","FAC","PRODUCT",
                "LANGUAGE","others"],
}

dataset_path = 'dataset/NER_data/'

def read_ner(datapath, domain):
    inputs, labels = [], []
    with open(datapath, "r", encoding="utf-8") as fr:
        token_list, label_list = [], []
        for i, line in enumerate(fr):
            line = line.strip()
            if domain in ['atis']:
                if line == "":
                    continue
                splits = line.split()
                assert(splits[0] == "BOS")
                sentence_end_idx = splits.index("EOS")
                sentence_splits = splits[1:sentence_end_idx]
                tag_splits = splits[sentence_end_idx+2:-1]
                assert(len(sentence_splits)==len(tag_splits))
                
                for token, label in zip(sentence_splits, tag_splits):
                    # tag_set.add(label)
                    if label not in domain2labels[domain]:
                        print('label %s not in label list.'%label)
                        label = 'O'
                        
                    subs_ = auto_tokenizer.tokenize(token)
                    if len(subs_) > 0:
                        label_list.extend([domain2labels[domain].index(label)] + [pad_token_label_id] * (len(subs_) - 1))
                        token_list.extend(auto_tokenizer.convert_tokens_to_ids(subs_))
                    else:
                        print("length of subwords for %s is zero; its label is %s" % (token, label))
                assert len(token_list) == len(label_list)
                inputs.append([auto_tokenizer.cls_token_id] + token_list + [auto_tokenizer.sep_token_id])
                labels.append([pad_token_label_id] + label_list + [pad_token_label_id])
                token_list, label_list = [], []
            else:
                if line == "":
                    if len(token_list) > 0:
                        assert len(token_list) == len(label_list)
                        inputs.append([auto_tokenizer.cls_token_id] + token_list + [auto_tokenizer.sep_token_id])
                        labels.append([pad_token_label_id] + label_list + [pad_token_label_id])

                    token_list, label_list = [], []
                    continue

                if domain in ['movie','restaurant']:
                    splits = line.split("\t")
                    token = splits[1]
                    label = splits[0]
                elif domain in ['conll2003','politics','science','music','literature','ai','PG','CG']:
                    splits = line.split("\t")
                    token = splits[0]
                    label = splits[1]
                elif domain in ['wnut17','ontonotes5']:
                    beg_idx = line.rfind(' ')
                    token = line[:beg_idx]
                    label = line[beg_idx+1:]

                subs_ = auto_tokenizer.tokenize(token)
                if len(subs_) > 0:
                    label_list.extend([domain2labels[domain].index(label)] + [pad_token_label_id] * (len(subs_) - 1))
                    token_list.extend(auto_tokenizer.convert_tokens_to_ids(subs_))
                else:
                    print("length of subwords for %s is zero; its label is %s" % (token, label))

    return inputs, labels


class Dataset(data.Dataset):
    def __init__(self, inputs, labels):
        self.X = inputs
        self.y = labels
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)

def collate_fn(data):
    X, y = zip(*data)
    lengths = [len(bs_x) for bs_x in X]
    max_lengths = max(lengths)
    padded_seqs = torch.LongTensor(len(X), max_lengths).fill_(auto_tokenizer.pad_token_id)
    padded_y = torch.LongTensor(len(X), max_lengths).fill_(pad_token_label_id)
    for i, (seq, y_) in enumerate(zip(X, y)):
        length = lengths[i]
        padded_seqs[i, :length] = torch.LongTensor(seq)
        padded_y[i, :length] = torch.LongTensor(y_)

    return padded_seqs, padded_y

def get_dataloader(data_path, domain, batch_size):

    inputs_train, labels_train = read_ner("%s%s/train.txt" %(data_path, domain), domain)
    inputs_dev, labels_dev = read_ner("%s%s/dev.txt" %(data_path, domain), domain)
    inputs_test, labels_test = read_ner("%s%s/test.txt" %(data_path, domain), domain)

    label_distri_dev = {}
    count_tok_test = 0
    for label_seq in labels_dev:
        for label in label_seq:
            if label != pad_token_label_id:
                label_name = domain2labels[domain][label]
                if "B-" in label_name or "S-" in label_name:
                    count_tok_test += 1
                    label_name = label_name.split("-")[1]
                    if label_name not in label_distri_dev:
                        label_distri_dev[label_name] = 1
                    else:
                        label_distri_dev[label_name] += 1
    #查看每个标签的概率分布
    for key in label_distri_dev:
        freq = label_distri_dev[key] / count_tok_test
        label_distri_dev[key] = round(freq, 5)

    label_distri_test = {}
    count_tok_test = 0
    for label_seq in labels_test:
        for label in label_seq:
            if label != pad_token_label_id:
                label_name = domain2labels[domain][label]
                if "B-" in label_name or "S-" in label_name:
                    count_tok_test += 1
                    label_name = label_name.split("-")[1]
                    if label_name not in label_distri_test:
                        label_distri_test[label_name] = 1
                    else:
                        label_distri_test[label_name] += 1

    for key in label_distri_test:
        freq = label_distri_test[key] / count_tok_test
        label_distri_test[key] = round(freq, 5)

    logger.info("train.txt size: %d; dev.txt size %d; test.txt size: %d;" % (len(inputs_train), len(inputs_dev), len(inputs_test)))

    label_mapping = get_label_mapping(domain)

    dataset_train = Dataset(inputs_train, labels_train)
    dataset_dev = Dataset(inputs_dev, labels_dev)
    dataset_test = Dataset(inputs_test, labels_test)
    
    dataloader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    dataloader_dev = DataLoader(dataset=dataset_dev, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return dataloader_train, dataloader_dev, dataloader_test, label_mapping


def get_label_mapping(domain):
    # mapping labels to class index (starts from zero)
    label2index = {label:index for index, label in enumerate(domain2labels[domain])}
    index2label = [label for index, label in enumerate(domain2labels[domain])]
    label_mapping = {"label2index":label2index, "index2label":index2label}
    return label_mapping

if __name__ == "__main__":
    pass
