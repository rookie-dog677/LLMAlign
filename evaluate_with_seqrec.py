import argparse

from seqrec.runner import Runner
from seqrec.utils import parse_command_line_args


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='SASRec', help='Model name, options: SASRec')
    parser.add_argument('--dataset', type=str, default='Games_5core', help='Source domain')
    parser.add_argument('--exp_type', type=str, default='srec')
    parser.add_argument('--embedding', type=str, default='', help='Path to item embedding .npy file')
    parser.add_argument('--seq_embedding', type=str, default='', help='whether pre-trained sequence embeddings are used')

    return parser.parse_known_args()


if __name__ == '__main__':
    args, unparsed_args = parse_args()
    command_line_configs = parse_command_line_args(unparsed_args)
    args_dict = vars(args)  

    merged_dict = {**args_dict, **command_line_configs}


    runner = Runner(
        model_name=args.model,
        config_dict= merged_dict
    )
    runner.run()

