import os.path
from labelling import is_malicious, load_attack_config
import csv
import multiprocessing
import subprocess
import numpy as np
import pandas as pd
from modelPipeline import ModelPipeline,FlowStreamer,ModelDataset
import argparse
import os
import torch
    
def clean_pcaps(source_dir, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"Created directory: {target_dir}")
    else:
        return

    for filename in os.listdir(source_dir):
        if filename.endswith(".pcap"):
            input_path = os.path.join(source_dir, filename)
            fname = filename.split('.')[0]
            output_path = os.path.join(target_dir, f"cleaned_{fname}.pcap")
            command = ["editcap", "-D", "10", input_path, output_path]
            
            try:
                subprocess.run(command, check=True)
                print(f"Success: {filename} -> cleaned_{fname}.pcap")
            except subprocess.CalledProcessError:
                print(f"Error: Failed to process {filename}")
            except FileNotFoundError:
                return
            

def label_pcap(pcap_file, labelled_data_dir, streamer, labelling_config=None):
    fname = os.path.basename(pcap_file).split(".")[0]
    out_filepath = f"{labelled_data_dir}/{fname}.csv"
    if labelling_config:
        timezone_offset, conf = labelling_config
    
    with open(out_filepath, mode='w', newline='') as csv_file:
        writer = None 
        
        for flow in streamer(pcap_file):
            if labelling_config:
                label = is_malicious(flow, timezone_offset, conf)
            else:
                label = 0
            
            for pkt in flow.udps.packets:
                if writer is None:
                    headers = list(pkt._fields) + ["flow_id", "label"]
                    
                    writer = csv.writer(csv_file)
                    writer.writerow(headers)
                    
                row_data = list(pkt) + [flow.id, label]
                writer.writerow(row_data)
                
    print(f"Labelled: {out_filepath}")


def label_data(streamer, data_dir, clean_data_dir, labelled_data_dir, labelling_config):
    clean_pcaps(data_dir, clean_data_dir)
    label_conf = load_attack_config(labelling_config)
    print("Starting NFStream...")
    for pcap_file in os.listdir(clean_data_dir):
        p = multiprocessing.Process(
            target=label_pcap, 
            args=(f"{clean_data_dir}/{pcap_file}", labelled_data_dir, label_conf, streamer)
        )
        p.start()
        p.join()

def get_train_test_mask(data, train_ratio, attack_thr_train_ratio):
    train_sample = set()
    test_sample = set()

    for i in data['label'].unique():
        group = data[data['label'] == i]['flow_id'].unique()
        eval_sample = np.random.choice(group, len(group), replace=False)
        benign_ratio = round(train_ratio*len(eval_sample))
        attack_ratio = round(attack_thr_train_ratio*len(eval_sample))
        if i == 0:
            train_sample |= (set(eval_sample[:benign_ratio]))
            test_sample |= (set(eval_sample[benign_ratio:]))
        elif i != 0:
            test_sample |= (set(eval_sample[:attack_ratio]))


    return np.array(list(train_sample)), np.array(list(test_sample))

def concat_dfs(labelled_data_dir):
    df_list = []
    current_max_flow_id = 0
    
    for idx, fname in enumerate(os.listdir(labelled_data_dir)):
        dfn = pd.read_csv(f"{labelled_data_dir}/{fname}")
        dfn.loc[dfn['label'] != 0, 'label'] = 1
        
        if idx > 0:
            dfn['flow_id'] = dfn['flow_id'] + current_max_flow_id + 1
            
        current_max_flow_id = dfn['flow_id'].max()
        df_list.append(dfn)

    df = pd.concat(df_list, ignore_index=True)
    return df


def train_test_split(labelled_data_dir, benign_train_ratio, attack_thr_train_ratio):

    df = concat_dfs(labelled_data_dir)

    df.reset_index(drop=True,inplace=True)
    train_idx,test_idx = get_train_test_mask(df, benign_train_ratio, attack_thr_train_ratio)

    df.index.name = 'index'
    df.sort_values(by=['flow_id', 'index'],inplace=True)
    df.set_index('flow_id', inplace=True)

    if len(train_idx) > 0:
        train_df = df[df.index.isin(train_idx)].copy()
        train_model_df = ModelDataset(train_df)
    else:
        train_model_df = None
    
    if len(test_idx) > 0:
        test_df = df[df.index.isin(test_idx)].copy()
        test_model_df = ModelDataset(test_df)
    else:
        test_model_df = None

    return train_model_df, test_model_df




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Pipeline")
    
    parser.add_argument('--data_dir', type=str, default='data', 
                        help='Directory for raw data')
    parser.add_argument('--clean_data_dir', type=str, default='clean_data', 
                        help='Directory for clean data')
    parser.add_argument('--labelled_data_dir', type=str, default='labelled_data', 
                        help='Directory for labelled data')
    parser.add_argument('--labelling_config', type=str, default='cicids2017_config.json', 
                        help='Labelling configuration')
    
    parser.add_argument('--eval', action='store_true', 
                        help='Set this flag to run to evaluate')
    parser.add_argument('--eval_thresh', action='store_true', 
                        help='Set this flag to evaluate thresholds')
    
    parser.add_argument('--rnd_seed', type=int, default=0, 
                        help='Random seed for reproducibility')
    parser.add_argument('--seq_len', type=int, default=32, 
                        help='Sequence length for FlowStreamer')
    parser.add_argument('--max_pkt_size', type=int, default=1460, 
                        help='Maximum packet size for FlowStreamer')
    parser.add_argument('--flow_timeout', type=int, default=120, 
                        help='Flow timeout for FlowStreamer in seconds')
    parser.add_argument('--model_path', type=str, default='model.pt', 
                        help='Path to save or load the model')
    
    parser.add_argument('--benign_train_ratio', type=float, default=0.6, 
                        help='Ratio of benign data used for training')
    parser.add_argument('--attack_thr_train_ratio', type=float, default=1.0, 
                        help='Ratio of attack threshold data used for unsupervised evaluation and threshold training')

    parser.add_argument('--embed_dim', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--augment_rate', type=float, default=0.4, help='Data augmentation rate')
    parser.add_argument('--temperature', type=float, default=0.5, help='Softmax temperature')
    parser.add_argument('--batch_size', type=int, default=128, help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device chosen:", device)
    
    np.random.seed(args.rnd_seed)
    torch.manual_seed(args.rnd_seed)
    if device == 'cuda':
        torch.cuda.manual_seed(args.rnd_seed)
    
    if args.eval:
        train_df, test_df = train_test_split(args.labelled_data_dir, 
                                             benign_train_ratio=args.benign_train_ratio, 
                                             attack_thr_train_ratio=args.attack_thr_train_ratio)
        pipeline = ModelPipeline.load(args.model_path, device)
        pipeline.eval(test_df)
        
    elif args.eval_thresh:
        train_df, test_df = train_test_split(args.labelled_data_dir, 
                                             benign_train_ratio=0, 
                                             attack_thr_train_ratio=args.attack_thr_train_ratio)
        pipeline = ModelPipeline.load(args.model_path, device)
        pipeline.eval(test_df, eval_threshold=True)
        
    else:
        streamer = FlowStreamer(seq_len=args.seq_len, 
                                max_pkt_size=args.max_pkt_size, 
                                flow_timeout=args.flow_timeout)
        
        if not os.path.exists(args.labelled_data_dir):
            os.makedirs(args.labelled_data_dir)
            label_data(streamer.stream_flows, 
                       data_dir=args.data_dir, 
                       clean_data_dir=args.clean_data_dir, 
                       labelled_data_dir=args.labelled_data_dir,
                       labelling_config=args.labelling_config)
        
        train_df, test_df = train_test_split(args.labelled_data_dir, 
                                             benign_train_ratio=args.benign_train_ratio, 
                                             attack_thr_train_ratio=args.attack_thr_train_ratio)
        
        pipeline = ModelPipeline(embed_dim=args.embed_dim,
                                 augment_rate=args.augment_rate,
                                 temperature=args.temperature,
                                 batch_size=args.batch_size, 
                                 device=device,
                                 learning_rate=args.learning_rate,
                                 streamer=streamer)
        
        pipeline.train_eval_save(train_df, test_df, args.model_path)
