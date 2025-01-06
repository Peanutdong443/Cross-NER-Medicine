import json
from src.config import *
from src.utils import *
from src.dataloader import *
from src.trainer import *
from src.model import *
from src.utils import *
import copy
import torch
import math
from tqdm import tqdm
from src.dataloader import domain2labels
import transformers


from transformers import (default_data_collator,get_scheduler)
from datasets import  load_dataset
import datasets

def train(params):
    # initialize experiment
    logger = init_experiment(params, logger_filename=params.logger_filename)
    from accelerate import Accelerator
    accelerator = Accelerator()

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    data_files = {}
    if params.tgt_dm is not None:
        data_files["train.txt"] = "dataset/ner_data/"+params.tgt_dm+"/train.json"
    if params.tgt_dm is not None:
        data_files["validation"] = "dataset/ner_data/"+params.tgt_dm+"/test.json"
    if params.tgt_dm is not None:
        data_files["dev.txt"] = "dataset/ner_data/"+params.tgt_dm+"/test.json"
    extension = "train.txt.json".split(".")[-1]
    raw_datasets = load_dataset(extension, data_files=data_files)

    if raw_datasets["train.txt"] is not None:
        column_names = raw_datasets["train.txt"].column_names
    else:
        column_names = raw_datasets["validation"].column_names
    text_column_name = column_names[0]
    label_column_name = column_names[1]
    if params.label_map_path is not None:
        ori_label_token_map = json.load(open(params.label_map_path, 'r'))
    else:
        ori_label_token_map = get_label_prompt(params.prompt_name)

    if params.label_schema == "BIO":
            new_ori_label_token_map = {}
            for key, value in ori_label_token_map.items():
                new_ori_label_token_map[key] = value
                new_ori_label_token_map["B-" + key[2:]] = value
            ori_label_token_map = new_ori_label_token_map

    label_list = list(ori_label_token_map.keys())
    #label_list=domain2labels[params.tgt_dm]
    label_list += 'O'

    print(len(label_list))
    label_to_id = {"O": 0}
    for l in label_list:
        if l != "O":
            label_to_id[l] = len(label_to_id)


    if True:  # args.eval_label_schema == "BIO":

        new_label_to_id = copy.deepcopy(label_to_id)
        for label, id in label_to_id.items():
            if label != "O" and "B-" + label[2:] not in label_to_id:
                new_label_to_id["B-" + label[2:]] = len(new_label_to_id)
        label_to_id = new_label_to_id
    id_to_label = {id: label for label, id in label_to_id.items()}
    source_dataloader_train, source_dataloader_dev, source_dataloader_test,source_mapping=get_dataloader(dataset_path,params.src_dm,32)
    target_dataloader_train, target_dataloader_dev, target_dataloader_test, target_mapping = get_dataloader(dataset_path,
                                                                                            params.tgt_dm,
                                                                                            params.batch_size_target)


    # Source domain NER Tagger
    model = BertTagger(params.src_dm,
                       params.tgt_dm,
                       params.hidden_dim,
                       params.model_name,
                       params.ckpt)
    model.cuda()

    tokenizer = AutoTokenizer.from_pretrained("Bert")
    tokenizer = add_label_token_bert(model.maskencoder, tokenizer, ori_label_token_map)
    label_token_map = {item: item for item in ori_label_token_map}

    label_token_to_id = {label: tokenizer.convert_tokens_to_ids(label_token) for label, label_token in
                         label_token_map.items()}
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            max_length=params.max_length,
            padding=False,
            truncation=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )

        target_tokens = []
        labels = []
        for i, label in enumerate(examples[label_column_name]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            input_ids = tokenized_inputs.input_ids[i]
            previous_word_idx = None
            label_ids = []
            target_token = []

            for input_idx, word_idx in zip(input_ids, word_ids):
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.

                if word_idx is None:
                    target_token.append(-100)
                    label_ids.append(-100)
                # Set target token for the first token of each word.
                elif word_idx != previous_word_idx:

                    label_ids.append(label_to_id[label[word_idx]])
                    if params.label_schema == "IO" and label[word_idx] != "O":
                        label[word_idx] = "I-" + label[word_idx][2:]

                    if label[word_idx] != 'O':
                        target_token.append(label_token_to_id[label[word_idx]])
                    else:
                        target_token.append(input_idx)
                    # target_tokens.append()

                # Set target token for other tokens of each word.
                else:
                    label_ids.append(label_to_id[label[word_idx]] if params.label_all_tokens else -100)
                    if params.label_schema == "IO" and label[word_idx] != "O":
                        label[word_idx] = "I-" + label[word_idx][2:]

                    if label[word_idx] != 'O':
                        # Set the same target token for each tokens.
                        target_token.append(label_token_to_id[label[word_idx]])

                        # Ignore the other words during training.
                        # target_token.append(-100)
                    else:
                        target_token.append(input_idx)
                previous_word_idx = word_idx
            target_tokens.append(target_token)
            labels.append(label_ids)

        tokenized_inputs["labels"] = target_tokens
        tokenized_inputs['ori_labels'] = labels
        # tokenized_inputs["poss"]=poss
        return tokenized_inputs

    processed_raw_datasets = raw_datasets.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=raw_datasets["train.txt"].column_names,
        desc="Running tokenizer on dataset",
    )
    train_dataset = processed_raw_datasets["train.txt"]
    eval_dataset = processed_raw_datasets["validation"]
    dev_dataset = processed_raw_datasets["dev.txt"]
    if params.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorForLMTokanClassification(
            tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None)
        )
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=params.batch_size_target
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=params.batch_size_target)
    dev_dataloader = DataLoader(dev_dataset, collate_fn=data_collator, batch_size=params.batch_size_target)
    trainer = BaseTrainer(params, model, tokenizer,label_mapping=(source_mapping, target_mapping))
    print(ori_label_token_map)

    # load the source model
    # trainer.load_model("source_model.pth", path=params.path_pretrain, is_source_model=True)
    logger.info("Training on target domain : %s ..."%(params.tgt_dm))
    trainer.set_source(False)
    trainer.set_optimizer()
    device = accelerator.device
    model.to(device)
    trainer.model, trainer.optimizer, train_dataloader, eval_dataloader, dev_dataloader = accelerator.prepare(
        trainer.model, trainer.optimizer, train_dataloader, eval_dataloader, dev_dataloader
    )
    # 创建源域图
    if params.load_source_graph:
        trainer.load_source_graph()
    else:
        trainer.set_source_graph(train_dataloader,
                                 is_save=True)


    #trainer.set_source_graph1(train_dataloader,is_save=True)
    if params.max_train_steps is None:
        params.max_train_steps = params.epoch_target * len(train_dataloader)
    else:
        params.num_train_epochs = math.ceil(params.max_train_steps / len(train_dataloader))
    lr_scheduler = get_scheduler(
        name=params.lr_scheduler_type,
        optimizer=trainer.optimizer,
        num_warmup_steps=params.num_warmup_steps,
        num_training_steps=params.max_train_steps,
    )
    progress_bar = tqdm(range(params.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    best_dev = 0
    best_f1 = 0
    for epoch in range(params.epoch_target):
        for step, batch in enumerate(train_dataloader):

            total_loss ,prompt_loss=trainer.train_step_prompt(batch)
            accelerator.backward(total_loss)

            if step % params.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                trainer.optimizer.step()
                lr_scheduler.step()
                trainer.optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            # if completed_steps >= params.max_train_steps:
            #     break

        # Test each epoch and save the model of the best epoch.
        # if epoch>=0:
            # best_metric = evaluate(best_metric)
        if not params.debug:
                # f1_dev = trainer.evaluate(dev_dataloader)
                f1_test = trainer.evaluate(eval_dataloader)
                # if (best_dev < f1_dev):
                #     best_dev = f1_dev
                #     best_f1 = f1_test
                # logger.info('f1_dev: %.4f'%f1_dev)
                logger.info('f1_test: %.4f' % f1_test)

        if epoch == params.epoch_target - 1:
            f1_test = trainer.evaluate(eval_dataloader)
            logger.info('f1_test: %.4f' % f1_test)
    logger.info('fininal_bers_f1_test: %.4f' % best_f1 )




if __name__ == "__main__":
    params = get_params()
    set_random_seed()
    train(params)
