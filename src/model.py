import dgl
from dgl.nn import GraphConv
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from transformers import AutoConfig
from transformers import AutoModelWithLMHead
from src.utils import *
from src.dataloader import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import logging
logger = logging.getLogger()
from transformers import AutoModelForMaskedLM
class BertTagger(nn.Module):
    def __init__(self, src_dm, tgt_dm, hidden_dim, model_name, ckpt):
        super(BertTagger, self).__init__()
        self.num_class_source = len(domain2labels[src_dm])
        self.num_class_target = len(domain2labels[tgt_dm])
        self.num_entity_class_target = len(domain2entitylabels[tgt_dm])
        self.hidden_dim = hidden_dim
        config = AutoConfig.from_pretrained(model_name)
        config.output_hidden_states = True
        self.encoder = AutoModelWithLMHead.from_pretrained(model_name, config=config)
        self.maskencoder = AutoModelForMaskedLM.from_pretrained(
            params.model_name,
            from_tf=bool(".ckpt" in params.model_name),
            config=config,
        ).cuda()
        if ckpt != '':
            logger.info("Reloading encoder from %s" % ckpt)
            encoder_ckpt = torch.load(ckpt)
            self.encoder.load_state_dict(encoder_ckpt)

        # The classifier
        self.is_source = True
        self.linear_source = nn.Linear(self.hidden_dim, self.num_class_source)
        self.linear_target = nn.Linear(self.hidden_dim, self.num_class_target).cuda()
        self.linear_target_2 = nn.Linear(self.hidden_dim, self.num_class_target).cuda()
        self.linear_target_bce = nn.Linear(self.hidden_dim*self.num_entity_class_target, 
                                            self.num_entity_class_target).cuda()

        # The linear projection to label related embedding
        self.proj_layer = nn.Sequential(
                                nn.Linear(self.hidden_dim, self.hidden_dim)
                            ).cuda()
        self.label_repre = nn.Parameter(torch.zeros(self.num_entity_class_target,
                                                    self.hidden_dim),
                                                requires_grad=True)
        nn.init.normal_(self.label_repre, mean=0.0, std=0.1)

        #注释图结构
        self.gcn = GraphConv(self.hidden_dim,
                            self.hidden_dim,
                            norm='none',
                            weight=True,
                            bias=True).cuda()
        self.source_graph_gcn = None
        self.source_edges = None


        #prompt调整
        self.linear_source_prompt = nn.Linear(29024, self.num_class_source)
        self.linear_target_prompt = nn.Linear(self.hidden_dim, self.hidden_dim).cuda()

        #因果结构

        self.s_sent2ner = nn.Linear(300, self.hidden_dim).cuda()
        self.s_ent2ner = nn.Linear(self.hidden_dim, self.hidden_dim).cuda()
        self.emb = nn.Embedding(28996, 300).cuda()
        self.middle_mlp = nn.Linear(self.hidden_dim, self.hidden_dim).cuda()
    def calculatelogits_s(self, I, X, Z):
        logitsdict = {}
        i2rlogits = self.s_sent2ner(I)
        logitsdict['i2rlogits'] = i2rlogits
        # Z-->Y
        z2rlogits = self.s_ent2ner(Z)
        logitsdict['z2rlogits'] = z2rlogits
        # X-->Y

        x2rlogits = self.s_ent2ner(X)
        logitsdict['x2rlogits'] = x2rlogits
        logits = i2rlogits + x2rlogits + z2rlogits
        return logitsdict, logits

    def forward(self, batch, auxillary_task=False, return_hiddens=False):
        X = batch['input_ids'].cuda()
        poss = batch.pop('poss', 'not found ner_labels')
        # pos_embs = self.dropout_in(self.pos_emb(poss.cuda()))
        outputs_target = self.maskencoder(**batch)
        # outputs_target=outputs_target.logits
        outputs_target=outputs_target.hidden_states[-1]
        outputs_encoder = self.encoder(X)  # a tuple ((bsz,seq_len,hidden_dim), (bsz, hidden_dim))
        outputs_encoder = outputs_encoder[1][-1]  # (bsz, seq_len, hidden_dim)
        outputs=self.linear_target_prompt(outputs_encoder)
        o_output = self.middle_mlp(outputs_encoder)

        # emb = pos_embs
        if self.is_source:
            prediction=self.linear_source(o_output)
            if return_hiddens:
                return prediction, outputs
            else:
                return prediction
        else:
            batch_size = outputs.shape[0]
            outputs_label_repre = self.proj_layer(outputs)
            label_repre = nn.Parameter(torch.zeros(self.num_entity_class_target,
                                                   self.hidden_dim),
                                       requires_grad=True).cuda()
            label_attention_logits = torch.bmm(label_repre.expand((batch_size, -1, -1)),
                                               outputs_label_repre.transpose(1, 2))
            label_attention_logits = label_attention_logits.masked_fill(
                X.unsqueeze(1).expand_as(label_attention_logits) == 0,
                1e-9)
            label_attention = torch.nn.functional.softmax(label_attention_logits, dim=-1)
            label_semantic_repre = torch.bmm(label_attention, outputs_label_repre)

            # for gcn
            label_semantic_repre_2 = self.gcn(self.source_graph_gcn.to('cuda'),
                                              label_semantic_repre.transpose(0, 1),
                                              edge_weight=self.source_edges.cuda())
            label_semantic_repre_2 = label_semantic_repre_2.transpose(0, 1)
            # for BCE classifier
            sentence_pred_bce = self.linear_target_bce(label_semantic_repre_2.reshape((batch_size, -1)))
            # attention for each token
            token_attention_logits = torch.bmm(outputs_label_repre,
                                               label_repre.expand((batch_size, -1, -1)).transpose(1, 2))
            token_attention = torch.nn.functional.softmax(token_attention_logits, dim=-1)
            token_label_repre = torch.bmm(token_attention, label_semantic_repre_2)

            prediction_1 = self.linear_target(outputs)
            prediction_2 = self.linear_target_2(token_label_repre)

            prediction = prediction_1 + prediction_2

            if return_hiddens:
                if auxillary_task:
                    return prediction, token_attention_logits, sentence_pred_bce, outputs
                else:
                    return prediction, outputs
            else:
                if auxillary_task:
                    return prediction, token_attention_logits, sentence_pred_bce
                else:
                    return prediction
