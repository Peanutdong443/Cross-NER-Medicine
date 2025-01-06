
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import dgl
from dgl.nn import EdgeWeightNorm, GraphConv
from src.conll2002_metrics import *
from src.dataloader import *
from src.utils import *
from src.utils_OT import *

import os
import numpy as np
from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt
import logging
import random
from sklearn.metrics import confusion_matrix
from seqeval.metrics import f1_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logger = logging.getLogger()

class BaseTrainer(object):
    def __init__(self, params, model, tokenizer,label_mapping=None):
        self.params = params
        self.model = model
        self.tokenizer=tokenizer
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = None
        # self.lr_source=0.01
        self.lr_target = params.lr_target
        self.lr_source = params.lr_source
        self.source_graph = None
        self.source_graph1 = None
        self.source_labels = np.array(label_mapping[0]['index2label'])
        self.target_labels = np.array(label_mapping[1]['index2label'])

        self.source_entitylabels = np.array(domain2entitylabels[self.params.src_dm])
        self.target_entitylabels = np.array(domain2entitylabels[self.params.tgt_dm])

    def set_source(self, is_source):
        self.model.is_source = is_source

    def set_optimizer(self):
        no_decay = ["bias", "LayerNorm.weight"]
        # build scheduler and optimizer
        if self.model.is_source:
            params_group = [{'params':self.model.encoder.parameters(),
                                'lr':self.lr_source},
                            {'params':self.model.linear_target.parameters(),
                                'lr':self.lr_source}]
            self.optimizer = torch.optim.SGD(params_group,
                                            weight_decay=0.0005, 
                                            momentum=0.9)
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, 
                                                                milestones=eval(self.params.schedule_source), 
                                                                gamma=self.params.gamma)
        else:
            params_group = [
                            {'params':self.model.encoder.parameters(),
                                'lr':self.lr_target},
                            {'params':self.model.linear_target.parameters(),
                                'lr':self.lr_target},
                            {'params':self.model.linear_target_2.parameters(),
                                'lr':self.lr_target},
                            {'params':self.model.linear_target_bce.parameters(),
                                'lr':self.lr_target},
                            {'params':self.model.proj_layer.parameters(),
                                'lr':self.lr_target},
                            {'params':self.model.label_repre,
                                'lr':self.lr_target},
                            {'params':self.model.gcn.parameters(),
                                'lr':self.lr_target},
                            {'params':self.model.linear_source_prompt.parameters(),
                                'lr':self.lr_target  },
                            {'params': self.model.s_sent2ner.parameters(),
                                'lr': self.lr_target},
                            {'params': self.model.s_ent2ner.parameters(),
                                 'lr': self.lr_target},
                            {'params': self.model.middle_mlp.parameters(),
                                 'lr': self.lr_target},
                            {
                            "params": [p for n, p in self.model.maskencoder.named_parameters() if not any(nd in n for nd in no_decay)],
                            "weight_decay": 0,
                            },
                            {
                            "params": [p for n, p in self.model.maskencoder.named_parameters() if any(nd in n for nd in no_decay)],
                            "weight_decay": 0,
                            }

                            ]


            self.optimizer = torch.optim.AdamW(params_group,
                                            weight_decay=0.0005,
                                            lr=self.params.lr_target
                                            )



    def get_dist_matrix(self, G1, G2):
        """
        input:
            G1: C1*D matrix, the nodes of the target graph (finetune model)
            G2: C2*D matrix, the nodes of the source graph (pretrain model)
        return:
            The distance matrix: C1*C2 matrix
        """
        return euclidean_dist(G1, G2)


    def get_graph_matching_loss(self, GS_all, GT_all, target_seen_label_mask, is_plot=False):
        """
        input:
            GS_all: CT*CS distance matrix for source nodes
                CT is the number of entity label in target domain(10 for politics)
                CS is the number of entity label in source domain(5 for conll2003)
            
            GT_all: _CT*CT distance matrix for target nodes
                _CT is the number of seen entity label in target domain(<=10 for politics)
                CT is the number of entity label in target domain(10 for politics)

            target_seen_label_mask: C2 vector for target nodes represents if the entity
                                    appear in the batch data
                   
        return:
            The sum of graph matching loss
        """

        GS_all, GT_all = GS_all.cuda(), GT_all.cuda()
        # discard the "others" dimension in the graph
        GS, GT = GS_all[:-1,:-1], GT_all[:-1,:-1] 

        # check dimensions
        CT, CS = GS.size(0), GS.size(1)
        # assert(CT==GT.size(0))
        _CT = GT.size(0)
        
        # get two graph with normalized edge
        # GS_sub = GS[target_seen_label_mask[:-1],:]    
        GS_sub = GS[:,:]    
        GS_sub_dist = self.get_dist_matrix(GS_sub, GS_sub)
        GT_dist = self.get_dist_matrix(GT, GT)
        # rescale
        scale_factor_1 = torch.mean(GS_sub_dist)
        scale_factor_2 = torch.mean(GT_dist)
        GS_sub_dist = GS_sub_dist/scale_factor_1
        GT_dist = GT_dist/scale_factor_2

        gwd, transport_plan = GW_distance_torch_batch_uniform(GS_sub_dist.unsqueeze(0).transpose(2,1),
                                                GT_dist.unsqueeze(0).transpose(2,1),
                                                lamda=1e-1,
                                                iteration=5,
                                                OT_iteration=20)
        total_wd = gwd

        return total_wd.squeeze(0)

    def get_label_mapping(self):
        dic = {}
        target_label_list = [l.lower() for l in self.target_labels]
        for i, l in enumerate(self.source_labels):
            l = l.lower()
            if l in target_label_list:
                dic[i] = target_label_list.index(l)
        return dic

    def set_source_graph(self, target_dataloader, is_save=True, is_plot=False):
        '''
            calculate and save the source graph
        '''
        logger.info("Building graph on source domain : %s ..."%(self.params.src_dm))
        self.model.eval()


        # store the is_source status before change it
        is_source_temp = self.model.is_source
        self.set_source(True)

        # for target data
        pred_list = []
        y_list = []
        pbar = tqdm(enumerate(target_dataloader), total=len(target_dataloader))
        for i, batch in pbar:

            # concatenation
            y = batch['ori_labels']
            a = batch['ori_labels'].size(0)
            b = batch['ori_labels'].size(1)
            y = y.view(a*b)
            y_list.append(y.cpu().clone().detach()) # y is a list
            X = batch['input_ids'].cuda()
            ner_label = batch.pop('ori_labels', 'not found ner_labels')
            preds = self.model(batch)
            preds = preds.view(-1, preds.size(-1)) 
            pred_list.append(preds.cpu().clone().detach())
        pred_list = torch.cat(pred_list)   # (length, num_tag)
        # f'(Xt)=softmax(f.(Xt)/T)
        temperature_preds = add_temperature(pred_list, temperature=self.params.temperature)
        temperature_scores = torch.nn.functional.softmax(temperature_preds, dim=1)

        score_list = torch.nn.functional.softmax(pred_list, dim=1)
        y_list = torch.cat(y_list)
        temperature_scores, y_list=filter_masked_label(temperature_scores, y_list, [-100])
        # get entitylabels
        temperature_scores = map_matrix_to_entitylabels(temperature_scores,
                                                        self.source_labels,
                                                        self.source_entitylabels)
        y_list = map_vec_to_entitylabels(y_list,
                                        self.target_labels,
                                        self.target_entitylabels)
        # note that here the load data is from the target domain,
        # so the label_list should be target_labels
        source_graph, class_seen_mask = get_label_center_graph(temperature_scores, y_list,
                                                        label_list=self.target_entitylabels)
        # ensure the source graph must contain all entity target labels
        assert(class_seen_mask==[True for _ in self.target_entitylabels])

        self.source_graph = source_graph
        if is_save:
            saved_path = os.path.join(self.params.dump_path, "source_graph")
            torch.save(source_graph, saved_path)
            logger.info("Source graph has been saved to %s" % saved_path)

        # restore the is_source status
        self.set_source(is_source_temp)
        source_graph1, class_seen_mask = get_feature_center_graph(temperature_scores, y_list,
                                                                  label_list=self.target_entitylabels)

        self.source_graph1 = source_graph1
        # set the source graph for gcn in the model
        self.set_source_graph_for_gcn(source_graph)

    def set_source_graph1(self, target_dataloader, is_save=True, is_plot=False):
        '''
            calculate and save the source graph
        '''
        logger.info("Building graph on source domain : %s ..." % (self.params.src_dm))
        self.model.eval()

        # store the is_source status before change it
        is_source_temp = self.model.is_source
        self.set_source(True)

        # for target data
        pred_list = []
        y_list = []
        pbar = tqdm(enumerate(target_dataloader), total=len(target_dataloader))
        for i, batch in pbar:
            # concatenation
            y = batch['ori_labels']
            a = batch['ori_labels'].size(0)
            b = batch['ori_labels'].size(1)
            y = y.view(a * b)
            y_list.append(y.cpu().clone().detach())  # y is a list
            X = batch['input_ids'].cuda()
            ner_label = batch.pop('ori_labels', 'not found ner_labels')
            preds = self.model(batch)
            preds = preds.view(-1, preds.size(-1))
            pred_list.append(preds.cpu().clone().detach())
        pred_list = torch.cat(pred_list)  # (length, num_tag)
        # f'(Xt)=softmax(f.(Xt)/T)
        temperature_preds = add_temperature(pred_list, temperature=self.params.temperature)
        temperature_scores = torch.nn.functional.softmax(temperature_preds, dim=1)

        score_list = torch.nn.functional.softmax(pred_list, dim=1)
        y_list = torch.cat(y_list)
        temperature_scores, y_list = filter_masked_label(temperature_scores, y_list, [-100])
        # get entitylabels
        temperature_scores = map_matrix_to_entitylabels(temperature_scores,
                                                        self.source_labels,
                                                        self.source_entitylabels)

        source_graph1, class_seen_mask = get_feature_center_graph(temperature_scores, y_list,
                                                               label_list=self.target_entitylabels)

        self.source_graph1 = source_graph1
        # self.set_source_graph_for_gcn1(source_graph)

    def set_source_graph_for_gcn1(self, GS):

        len_GS=len(GS)
        # get euclidean distance matrix
        GS_dist = euclidean_dist(GS, GS)
        # rescale the distance matrix for edge weight
        GS_dist = GS_dist / torch.mean(GS_dist)
        edge_weight = GS_dist.flatten()
        out_nodes = [i for _ in range(len_GS)
                     for i in range(len_GS)]
        in_nodes = [j for j in range(len_GS)
                    for _ in range(len_GS)]
        _out_nodes = []
        _in_nodes = []
        _edge_weight_idx = []
        for i, (o_node, i_node, weight) in enumerate(zip(out_nodes, in_nodes, edge_weight)):
            if weight < 1.5:
                _out_nodes.append(o_node)
                _in_nodes.append(i_node)
                _edge_weight_idx.append(i)
        source_graph_gcn = dgl.graph((_out_nodes, _in_nodes))
        norm = EdgeWeightNorm(norm='both')
        norm_edge_weight = norm(source_graph_gcn, edge_weight[_edge_weight_idx])
        self.model.source_feature_edges = norm_edge_weight
        self.model.source_graph_feature_gcn = source_graph_gcn

    def set_source_graph_for_gcn(self, GS):

        # get euclidean distance matrix
        GS_dist = euclidean_dist(GS, GS)
        # rescale the distance matrix for edge weight
        GS_dist = GS_dist/torch.mean(GS_dist)
        edge_weight = GS_dist.flatten()
        out_nodes = [i for _ in range(len(self.target_entitylabels))
                     for i in range(len(self.target_entitylabels))]
        in_nodes = [j for j in range(len(self.target_entitylabels)) \
                                    for _ in range(len(self.target_entitylabels))]
        _out_nodes = []
        _in_nodes = []
        _edge_weight_idx = []
        for i, (o_node, i_node, weight) in enumerate(zip(out_nodes, in_nodes, edge_weight)):
            if weight < 1.5:
                _out_nodes.append(o_node)
                _in_nodes.append(i_node)
                _edge_weight_idx.append(i)    
        source_graph_gcn = dgl.graph((_out_nodes,_in_nodes))
        norm = EdgeWeightNorm(norm='both')
        norm_edge_weight = norm(source_graph_gcn, edge_weight[_edge_weight_idx])
        self.model.source_edges = norm_edge_weight
        self.model.source_graph_gcn = source_graph_gcn

    def load_source_graph(self):
        '''
            load the source graph from files
        '''
        load_path = os.path.join(self.params.dump_path, "source_graph")
        self.source_graph = torch.load(load_path)
        print(self.source_graph)
        logger.info("Source graph has been loaded from %s" % load_path)


    def train_step_prompt(self, batch, update_center=False, is_plot=False):
        self.model.train()
        y = batch.pop('ori_labels', 'not found ner_labels')
        if self.model.is_source:
            preds = self.model(batch)
        else:
            preds, atten_pred, sentence_pred= self.model(batch, auxillary_task=True)
        # supervise learning loss
        y_flatten = y.view(y.size(0) * y.size(1)).long()
        preds_flatten = preds.view(-1, preds.size(-1))
        learning_loss = self.loss_fn(preds_flatten, y_flatten)
        sentence_bce_loss = torch.tensor(0)
        prompt_loss=torch.tensor(0)
        t=0
        if not self.model.is_source:
            # prompt loss

            temperature_preds = add_temperature(preds_flatten, temperature=self.params.temperature)
            temperature_scores = torch.nn.functional.softmax(temperature_preds, dim=1)
            _y = y_flatten.clone().detach()
            temperature_scores, _y = filter_masked_label(temperature_scores, _y, [-100])
            temperature_scores = map_matrix_to_entitylabels(temperature_scores,
                                                            self.target_labels,
                                                            self.target_entitylabels)
            _y = map_vec_to_entitylabels(_y,
                                         self.target_labels,
                                         self.target_entitylabels)
            # target_subgraph, target_seen_label_mask = get_label_center_graph(
            #     temperature_scores, _y,
            #     label_list=self.target_entitylabels)
            target_subgraph, target_seen_label_mask = get_feature_center_graph(
                temperature_scores, _y,
                label_list=self.target_entitylabels)
            t=t+len(target_subgraph)
            soure_graph=self.source_graph1[:t,:]

            graph_matching_loss = self.get_graph_matching_loss(soure_graph,
                                                               target_subgraph,
                                                               target_seen_label_mask)
            y_entity_label = map_vec_to_entitylabels(y_flatten,
                                                     label_list=self.target_labels,
                                                     entitylabel_list=self.target_entitylabels).long()
            y_entity_label = y_entity_label.reshape_as(y)
            y_bce = torch.zeros_like(sentence_pred)
            for i in range(y_bce.shape[0]):
                for j in range(y_bce.shape[1]):
                    if j in y_entity_label[i]:
                        y_bce[i][j] = 1
            y_bce = y_bce.cuda()
            bce_loss_fn = nn.BCEWithLogitsLoss()
            sentence_bce_loss = bce_loss_fn(sentence_pred, y_bce)

        #total_loss = learning_loss + self.params.lambda_1 * sentence_bce_loss+self.params.lambda_1*graph_matching_loss
        total_loss = learning_loss + self.params.lambda_1 * sentence_bce_loss+self.params.lambda_2*graph_matching_loss
        #otal_loss = learning_loss

        return total_loss ,prompt_loss

    def evaluate(self, dataloader, is_plot=False):
        self.model.eval()

        hidden_list = []
        pred_list = []
        y_list = []

        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        
        for i, batch in pbar:
            try:
                # concatenation
                y = batch.pop('ori_labels', 'not found ner_labels')
                y = y.view(y.size(0)*y.size(1))
                y_list.append(y.cpu().detach()) # y is a list

                preds = self.model(batch)

                preds = preds.view(-1, preds.size(-1))
                pred_list.append(preds.cpu().detach())
            except:
                continue

        pred_list = torch.cat(pred_list)   # (length, num_tag)
        y_list = torch.cat(y_list)

        #####
        pred_list = torch.argmax(pred_list, dim=1)

        # calcuate f1 score
        pred_list = list(pred_list.numpy())
        y_list = list(y_list.numpy())

        pred_line = []
        gold_line = []
        for pred_index, gold_index in zip(pred_list, y_list):
            gold_index = int(gold_index)
            if gold_index != pad_token_label_id:
                if self.model.is_source:
                    pred_token = self.source_labels[pred_index]
                    gold_token = self.source_labels[gold_index]
                else:
                    pred_token = self.target_labels[pred_index]
                    gold_token = self.target_labels[gold_index]

                pred_line.append(pred_token) 
                gold_line.append(gold_token) 

        f1 = f1_score([gold_line], [pred_line])*100
        return f1

    def save_model(self, save_model_name, path=''):
        """
        save the best model
        """
        if len(path)>0:
            saved_path = os.path.join(path, str(save_model_name))
        else:
            saved_path = os.path.join(self.params.dump_path, str(save_model_name))
        torch.save({
            "model": self.model,
            "source_domain": str(self.params.src_dm),
            "target_domain": str(self.params.tgt_dm),
        }, saved_path)
        logger.info("Best model has been saved to %s" % saved_path)

    def load_model(self, load_model_name, path='', is_source_model=False):
        """
        load the checkpoint
        """
        if len(path)>0:
            load_path = os.path.join(path, str(load_model_name))
        else:
            load_path = os.path.join(self.params.dump_path, str(load_model_name))
        ckpt = torch.load(load_path)

        # ensure the domain and the classifier matches the current settings
        if is_source_model:
            # self.model.__init__(self.params.src_dm,
            #                     self.params.tgt_dm,
            #                     self.params.hidden_dim,
            #                     self.params.model_name,
            #                     self.params.ckpt)
            self.model.linear_source = ckpt['model'].linear_source.cuda()
            self.model.encoder = ckpt['model'].encoder.cuda()

            

        elif not is_source_model:
            assert(self.params.src_dm==ckpt.get('source_domain'))
            assert(self.params.tgt_dm==ckpt.get('target_domain'))
            self.model = ckpt['model']
            # self.model=ckpt
        logger.info("Model has been load from %s" % load_path)