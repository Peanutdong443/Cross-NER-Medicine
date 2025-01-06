import argparse
from transformers import SchedulerType
def get_params():
    # parse parameters
    parser = argparse.ArgumentParser(description="Cross-domain NER")
    parser.add_argument("--exp_name", type=str, default="default", help="Experiment name")
    parser.add_argument("--logger_filename", type=str, default="train.txt.log")

    parser.add_argument("--dump_path", type=str, default="experiments", help="Experiment saved root path")
    parser.add_argument("--exp_id", type=str, default="50", help="Experiment id")
    parser.add_argument("--model_name", type=str, default="Bert", help="model name (e.g., bert-base-cased, roberta-base or wide_resnet)")
    parser.add_argument("--ckpt", type=str, default="", help="pretrain LM")
    parser.add_argument("--src_dm", type=str, default="conll2003", help="source domain")
    parser.add_argument("--tgt_dm", type=str, default="CG", help="target domain")
    parser.add_argument("--prompt_name", type=str, default="CG", help="target domain prompt")


    parser.add_argument("--debug", default=False, action="store_true", help="if skipping the test.txt on training and validation set")
    parser.add_argument("--test_pretrain", default=True, action="store_true", help="test.txt on the pretrain model")
    parser.add_argument("--test_finetune", default=True, action="store_true", help="test.txt on the finetune model")
    parser.add_argument("--path_pretrain", type=str, default='model', help="path to load/save pretrain (source) model")
    parser.add_argument("--path_finetune", type=str, default='', help="path to load/save finetune model")

    parser.add_argument("--is_set_source", default=True, action="store_true", help="load the source graph")

    parser.add_argument("--load_source_graph", default=False, action="store_true", help="load the source graph")
    parser.add_argument("--load_label_mapping", default=False, action="store_true", help="load the label mapping")
    parser.add_argument("--lr_source", type=float, default=0.1, help="Learning rate")
    # train.txt parameters
    parser.add_argument("--epoch_target", type=int, default=20, help="Number of epoch in target domain")
    parser.add_argument("--batch_size_target", type=int, default=8, help="Batch size in target domain")
    parser.add_argument("--lr_target", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--schedule_source", type=str, default='(4, 7)', help="Multistep scheduler")
    parser.add_argument("--schedule_target", type=str, default='(30, 60)', help="Multistep scheduler")
    parser.add_argument("--gamma", type=float, default=0.2, help="Factor of the learning rate decay")
    parser.add_argument("--hidden_dim", type=int, default=768, help="Hidden layer dimension")

    parser.add_argument("--lambda_1",type=float, default=0.1, help="weight of sentence_bce_loss")
    parser.add_argument("--lambda_2",type=float, default=0.01, help="weight of graph_matching_loss")
    parser.add_argument("--temperature",type=float, default=4, help="temperature for the score of building graph")



    parser.add_argument(
        "--label_map_path",
        default=None,
        type=str,
        help="label map path",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        default='pretrained/bert-base-cased'
    )

    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lenght` is passed."
        ),
    )
    parser.add_argument(
        "--label_schema",
        default='BIO',
        help="Activate debug mode and run training only with a subset of data.",
    )
    parser.add_argument(
        "--label_all_tokens",
        action="store_true",
        help="Setting labels of all special tokens to -100 and thus PyTorch will ignore them.",
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--eval_label_schema",
        default='BIO',
        help="Activate debug mode and run training only with a subset of data.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    params = parser.parse_args()

    return params
