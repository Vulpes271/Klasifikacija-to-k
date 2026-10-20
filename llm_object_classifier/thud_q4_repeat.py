"""Optional part-one Q4 text-only repeat on the existing 20 frozen captions.

Reuses the original stage2 runner with long-no-examples, T=.3, seed 42.
No image is opened, selected, or sent. Run after the main experiment and
regenerate the report; completed jobs are preserved by the shared launcher.
"""
import argparse
from .thud_experiment import DATASETS, MODELS, OUT, configuration, execute, progress


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url',default='http://172.18.0.2:11434')
    args=parser.parse_args()
    for dataset in DATASETS:
        frozen=OUT/'frozen'/dataset
        if len(list(frozen.glob('frame-*.json')))!=10:
            raise ValueError('Run the main experiment first; need 10 frozen captions per dataset.')
        execute(f'reason-q4-long-no-examples-t0.3-s42-{dataset}',
                configuration(MODELS[1],args.base_url,'long-no-examples',.3,42,2),2,frozen)
    progress('Dodatna besedilna Q4 ponovitev zaključena na istih 20 zamrznjenih opisih.')

if __name__=='__main__':main()
