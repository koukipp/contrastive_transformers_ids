from nfstream import NFStreamer
import torch
from packetCollector import PacketCollector, PacketData
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from model import BERT,NTXent
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


class ModelDataset:
    def __init__(self, df):
        count_col = df.index.map(df.index.value_counts())
        self.counts_indexes = {}
        for i in range(1, count_col.max() + 1):
            self.counts_indexes[i] = df[count_col == i].index.unique().to_numpy()

        self.flow_boundaries = {}
        self.unique_ids = df.index.unique()

        for flow_id in self.unique_ids:
            start = df.index.searchsorted(flow_id, side='left')
            end = df.index.searchsorted(flow_id, side='right')
            self.flow_boundaries[flow_id] = (start, end)
        

        self.dataset_np = df.to_numpy(dtype=np.float32)

class FlowStreamer():
    def __init__(self,
                seq_len,
                flow_timeout,
                max_pkt_size):
    
        self.seq_len = seq_len
        self.flow_timeout = flow_timeout
        self.max_pkt_size = max_pkt_size


    def process_flow(self, flow):
        flow.udps.packets = flow.udps.packets[:self.seq_len] 
        prev_time = None
        packets = []
        for pkt in flow.udps.packets:
            current_time = pkt.timestamp
            iat = (current_time - prev_time) if prev_time is not None else 0.0
            prev_time = current_time
            
            
            updated_pkt = PacketData(
                timestamp=round(iat, 3) / self.flow_timeout,
                size=pkt.size / self.max_pkt_size,
                direction=pkt.direction,
                ip_protocol=pkt.ip_protocol,
                tcp_flags=pkt.tcp_flags
            )

            packets.append(updated_pkt)
            
        flow.udps.packets = packets
        
        return flow

    def stream_flows(self, source):
        bpf_filter = "ip and (tcp or udp or icmp or igmp) and not broadcast"
        streamer = NFStreamer(source=source,bpf_filter=bpf_filter,
                                idle_timeout=self.flow_timeout, 
                                active_timeout=self.flow_timeout, 
                                udps=[PacketCollector()], 
                                decode_tunnels=True, 
                                accounting_mode=3, 
                                n_dissections=0,
                                n_meters=1)
        for flow in streamer:
            if (not flow.act_timeout # Take only the first timeout seconds for a long-living flow, ignore the rest
            and flow.bidirectional_bytes > 0 
            and flow.bidirectional_packets > 1 
            and (flow.protocol != 6 or flow.udps.packets[0].tcp_flags == 2)):
                yield self.process_flow(flow)

class ModelPipeline():
    def __init__(self,
                 device,
                 streamer,
                 embed_dim, 
                 augment_rate,
                 temperature,
                 batch_size,
                 learning_rate,
                 num_columns_names = ['timestamp','size']):
        
        
        self.device = device
        self.loss_fn = NTXent(temperature=temperature)
        self.augment_rate = augment_rate
        self.embed_dim = embed_dim
        self.streamer = streamer
        self.batch_size = batch_size
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.num_columns_names = num_columns_names

        self.num_features = len(PacketData._fields)
        self.num_columns = [PacketData._fields.index(col) for col in self.num_columns_names]
        self.epochs = 0
        self.benign_arr = None
        self.best_threshold = None

        self.model = BERT(seq_len=self.streamer.seq_len, num_features=self.num_features, num_columns=self.num_columns)
        self.model.to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), learning_rate)
    
    
    def save(self, filepath):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            
            'model_hparams': {
                'embed_dim': self.embed_dim,
                'augment_rate': self.augment_rate,
                'temperature': self.temperature,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'num_columns_names': self.num_columns_names
            },

            'streamer_hparams': {
                'seq_len': self.streamer.seq_len,
                'max_pkt_size': self.streamer.max_pkt_size,
                'flow_timeout': self.streamer.flow_timeout
            },
            
            'benign_arr': self.benign_arr,
            'best_threshold': self.best_threshold,
            'epochs': self.epochs
        }
        torch.save(checkpoint, filepath)
        print(f"Model successfully saved to {filepath}")

    @classmethod
    def load(cls, filepath, device):
        checkpoint = torch.load(filepath, map_location=device)
        
        s_hparams = checkpoint['streamer_hparams']
        fresh_streamer = FlowStreamer(**s_hparams)
        model_kwargs = checkpoint['model_hparams']
            
        instance = cls(device=device, streamer=fresh_streamer, **model_kwargs)
        
        instance.model.load_state_dict(checkpoint['model_state_dict'])
        instance.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        instance.epochs = checkpoint['epochs']
        instance.benign_arr = checkpoint['benign_arr']
        instance.best_threshold = checkpoint['best_threshold']
        
        print(f"Model successfully loaded from {filepath}")
        return instance
    
    def generate_batches(self, indexes):
        shuffled_indexes = np.random.permutation(indexes)
        for i in range(0, len(shuffled_indexes), self.batch_size):
            yield shuffled_indexes[i : i + self.batch_size]
    
    @staticmethod
    def pad_and_mask(tensor_list):
        device = tensor_list[0].device
            
        lengths = torch.tensor([t.size(0) for t in tensor_list], device=device)
        max_len = lengths.max()
        batch_size = len(tensor_list)
        
        padded_tensor = torch.nn.utils.rnn.pad_sequence(
            tensor_list, batch_first=True, padding_value=0.0
        )
        
        arange = torch.arange(max_len, device=device).expand(batch_size, max_len)
        padding_mask = arange >= lengths.unsqueeze(1)
        
        return padded_tensor, padding_mask
    
    def batch_loader(self, df, augment_rate=None):
        for batch in self.generate_batches(df.unique_ids):
            tensor_x_list = []
            tensor_y_list = []
            
            for flow_id in batch:
                start, end = df.flow_boundaries[flow_id]
                
                group_np = df.dataset_np[start : end] 
                seq_x = group_np[:, :-1]
                seq_len = seq_x.shape[0]
                
                tensor_x_list.append(torch.from_numpy(seq_x))

                if augment_rate:
                    seq_x_aug = seq_x.copy()
                    if seq_len > 1:

                        rand_flow_id = np.random.choice(df.counts_indexes[seq_len])
                        new_start, new_end = df.flow_boundaries[rand_flow_id]
                        new_seq = df.dataset_np[new_start : new_end]
                        
                        if new_seq.ndim == 1:
                            new_seq = new_seq.reshape(1, -1)
                        new_seq = new_seq[:, :-1]
                        
                        cut_length = round(augment_rate * seq_len)
                        start_a = np.random.randint(0, seq_len - cut_length + 1)
                        seq_x_aug[start_a:start_a + cut_length] = new_seq[start_a:start_a + cut_length]
                    
                    tensor_y_list.append(torch.from_numpy(seq_x_aug))
                else:
                    tensor_y_list.append(group_np[0, -1])

            tensor_x, padding_mask = ModelPipeline.pad_and_mask(tensor_x_list)
            
            if augment_rate:
                tensor_y = torch.nn.utils.rnn.pad_sequence(tensor_y_list, batch_first=True, padding_value=0.0)
            else:
                tensor_y = torch.tensor(tensor_y_list, device=self.device)

            yield tensor_x.to(self.device), tensor_y.to(self.device), padding_mask.to(self.device)
    
    def get_similarity(self, input, mask):
        y_out = F.normalize(self.model.embeddings(input, mask), p=2, dim=1)
        similarity = torch.mm(y_out, self.benign_arr)
        cos_sim, _ = similarity.max(dim=1)
        return cos_sim

    def eval(self, df, eval_threshold=False, unsup_thresh=5e-3):
        if not self.epochs:
            print("Model not trained")
            return
        
        if not df:
            print("No evaluation data")
            return

        self.model.eval()

        total_samples = len(df.unique_ids) 
        y_test = torch.empty(total_samples, dtype=torch.long)
        cos_sim_out = torch.empty(total_samples, dtype=torch.float32)

        current_idx = 0

        with torch.no_grad():
            for tensor_x_test, tensor_y_test, mask in self.batch_loader(df):
                
                current_batch_size = tensor_x_test.shape[0]

                cos_sim = self.get_similarity(tensor_x_test, mask)
                
                y_test[current_idx : current_idx + current_batch_size] = tensor_y_test
                cos_sim_out[current_idx : current_idx + current_batch_size] = cos_sim
                
                current_idx += current_batch_size
        
        if not eval_threshold:
            if torch.any(y_test):
                roc_score = roc_auc_score(y_test, 1 - cos_sim_out)
                fpr, tpr, thresholds = roc_curve(y_test, 1 - cos_sim_out)
                youden_j = tpr - fpr
                best_threshold_index = np.argmax(youden_j)
                best_threshold = float(thresholds[best_threshold_index])
                self.best_threshold = best_threshold


                print(accuracy_score(y_test, (1 - cos_sim_out) >= best_threshold))
                print(classification_report(y_test, (1 - cos_sim_out) >= best_threshold,digits=4))
                print("Best Threshold:", best_threshold)
                print("TPR:", tpr[best_threshold_index], "FPR:", fpr[best_threshold_index])
                print("ROC_AUC_SCORE", roc_score)
            else:
                self.best_threshold = float(torch.quantile(1 - cos_sim_out, 1 - unsup_thresh))
                print("Best Threshold (without attack samples):", self.best_threshold)

        elif self.best_threshold:
            best_threshold = self.best_threshold
            roc_score = roc_auc_score(y_test, 1 - cos_sim_out)
            print(accuracy_score(y_test, (1 - cos_sim_out) >= best_threshold))
            print(classification_report(y_test, (1 - cos_sim_out) >= best_threshold,digits=4))
            print("ROC_AUC_SCORE", roc_score)

    
        
    def train_eval_save(self, train_df, test_df=None, save_name=None):
        if not train_df:
            print("No training data")
            return
        running_loss = 0.

        self.model.train(True)
        batch_num = len(train_df.unique_ids) // self.batch_size
        for i, batch in enumerate(self.batch_loader(train_df, self.augment_rate)):
            loss = self.train_batch(batch)
            running_loss += loss.detach()

            if i % 100 == 0:
                print(f"batch: {i}/{batch_num} loss: {running_loss.item() / (i + 1)}")
        
        last_loss = running_loss / (i+1)
        print('epoch {} batches {} loss: {}'.format(self.epochs, i + 1, last_loss))
        self.epochs += 1

        self.model.eval()

        print("Creating benign embeddings...")
        total_samples = len(train_df.unique_ids)
        self.benign_arr = torch.empty((total_samples, self.embed_dim), device=self.device)

        current_idx = 0
        with torch.no_grad():
            for i, batch in enumerate(self.batch_loader(train_df)):
                tensor_x_test, _, mask = batch
                        
                y_out = self.model.embeddings(tensor_x_test, mask)
                
                current_batch_size = y_out.shape[0]
                
                self.benign_arr[current_idx : current_idx + current_batch_size] = y_out
                
                current_idx += current_batch_size
            
        self.benign_arr = F.normalize(self.benign_arr, p=2, dim=1).t()

        if test_df:
            self.eval(test_df)
        
        if save_name:
            print("Saving model...")
            self.save(save_name)
            
    def train_batch(self, batch):
        self.optimizer.zero_grad()
        Xbatch, xbatch_pair, mask = batch
        anchor, pair = self.model(Xbatch, xbatch_pair, mask)
        loss = self.loss_fn(anchor, pair)
        loss.backward()
        self.optimizer.step()
        return loss