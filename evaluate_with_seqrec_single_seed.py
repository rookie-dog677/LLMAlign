import argparse
import json
import os

from seqrec.runner import Runner
from seqrec.utils import parse_command_line_args


def make_json_safe(value):
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='SASRec', help='Model name, options: SASRec')
    parser.add_argument('--dataset', type=str, default='Games_5core', help='Dataset name')
    parser.add_argument('--exp_type', type=str, default='srec')
    parser.add_argument('--embedding', type=str, default='', help='Item embedding path')
    parser.add_argument('--seq_embedding', type=str, default='', help='Sequence embedding path pattern')
    parser.add_argument('--rand_seed', type=int, default=2024, help='Random seed')
    parser.add_argument('--out_json', type=str, required=True, help='Where to save evaluation payload')
    return parser.parse_known_args()


if __name__ == '__main__':
    args, unparsed_args = parse_args()
    command_line_configs = parse_command_line_args(unparsed_args)
    args_dict = vars(args)
    merged_dict = {**args_dict, **command_line_configs}

    runner = Runner(
        model_name=args.model,
        config_dict=merged_dict,
    )
    test_result, exp_config = runner.run()

    payload = {
        'seed': int(args.rand_seed),
        'test_result': make_json_safe(test_result),
        'exp_config': make_json_safe(exp_config),
        'merged_dict': make_json_safe(merged_dict),
    }

    out_dir = os.path.dirname(os.path.abspath(args.out_json))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out_json, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))
