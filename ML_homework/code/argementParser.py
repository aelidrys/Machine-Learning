import argparse

# Create the parser
parser = argparse.ArgumentParser(description="A simple argument parser example")

parser.add_argument('--dataset', type=str, default='hWork2.csv')
parser.add_argument('--preprocessing', type=int, default=1,
    help='0 for no processing, 1 for min/max, 2 for standrizing')

args = parser.parse_args()
print(f'data_set: {args.dataset} | processe: {args.preprocessing}')

