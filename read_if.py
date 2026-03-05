
import torch
import sys
import queue
import threading
import time
from modelPipeline import ModelPipeline

BATCH_SIZE = 128
TIMEOUT = 2.0  

flow_queue = queue.Queue()

def flow_reader(streamer, iface):
    for flow in streamer.stream_flows(iface):
        if flow.src_ip != '127.0.0.1':
            tensor = torch.tensor(flow.udps.packets, dtype=torch.float32)
            flow_queue.put((flow, tensor))


def main_loop(pipeline, threshold, device):
    buffer_flows = []
    buffer_tensors = []
    last_batch_time = time.time()
    flagged_ips = {}
    banned_ips = set()
    pipeline.model.eval()

    while True:
        try:
            now = time.time()
            # Wait at most TIMEOUT seconds for a new flow
            wait_time = max(0.0, TIMEOUT - (now - last_batch_time))
            try:
                flow, tensor = flow_queue.get(timeout=wait_time)
                buffer_flows.append(flow)
                buffer_tensors.append(tensor)
            except queue.Empty:
                pass

            # Check batch conditions
            if len(buffer_tensors) >= BATCH_SIZE or (buffer_tensors and (time.time() - last_batch_time > TIMEOUT)):
                batched_flows, mask = ModelPipeline.pad_and_mask(buffer_tensors)

                with torch.no_grad():
                    print("Processing",len(buffer_tensors),"flows")
                    cos_sim = pipeline.get_similarity(batched_flows.to(device), mask.to(device))
                    flow_idx, = torch.where(cos_sim < threshold)
                    flow_idx_norm, = torch.where(cos_sim >= threshold)

                    curr_batch_flagged = set()

                    for idx in flow_idx.tolist():
                        flow = buffer_flows[idx]
                        if flow.src_ip in flagged_ips:
                            if flow.bidirectional_first_seen_ms - flagged_ips[flow.src_ip]['time'] < (3 * 120 * 10**6):
                                flagged_ips[flow.src_ip]['cnt'] = flagged_ips[flow.src_ip]['cnt'] + 1
                                flagged_ips[flow.src_ip]['anom_score_sum'] = flagged_ips[flow.src_ip]['anom_score_sum'] + cos_sim[idx]
                                flagged_ips[flow.src_ip]['flagged_flows'].append(flow)
                            else:
                                flagged_ips[flow.src_ip]['time'] = flow.bidirectional_first_seen_ms
                                flagged_ips[flow.src_ip]['cnt'] = 1
                                flagged_ips[flow.src_ip]['cnt_normal'] = 0
                                flagged_ips[flow.src_ip]['anom_score_sum'] = cos_sim[idx]
                                flagged_ips[flow.src_ip]['flagged_flows'] = [flow]
                        else:
                              flagged_ips[flow.src_ip] = {'time': flow.bidirectional_first_seen_ms, 'cnt': 1, 'cnt_normal':0, 'anom_score_sum': cos_sim[idx], 'flagged_flows': [flow]}

                        curr_batch_flagged.add(flow.src_ip)
                    
                    for idx in flow_idx_norm:
                        flow = buffer_flows[idx]
                        if flow.src_ip in flagged_ips:
                            flagged_ips[flow.src_ip]['cnt_normal'] = flagged_ips[flow.src_ip]['cnt_normal'] + 1
                            flagged_ips[flow.src_ip]['anom_score_sum'] = flagged_ips[flow.src_ip]['anom_score_sum'] + cos_sim[idx]
                    
                    for src_ip in curr_batch_flagged:
                        sum_cnt = flagged_ips[src_ip]['cnt'] + flagged_ips[src_ip]['cnt_normal']
                        avg_score = flagged_ips[src_ip]['anom_score_sum'] / sum_cnt
                        
                        if flagged_ips[src_ip]['cnt'] >= 3 and (avg_score < threshold) and src_ip not in banned_ips:
                            print(f"Malicious ip:{src_ip},{avg_score},{flagged_ips[src_ip]['cnt']},{sum_cnt}")
                            print("Flagged flows", flagged_ips[src_ip]['flagged_flows'])
                            for flow in flagged_ips[src_ip]['flagged_flows']:
                                print(flow.src_ip,flow.dst_ip,flow.src_port,flow.dst_port,flow.udps.packets)

                            banned_ips.add(src_ip)

                print(f"[Main Loop] Processed batch of {len(buffer_tensors)} flows")

                buffer_flows.clear()
                buffer_tensors.clear()
                last_batch_time = time.time()

        except KeyboardInterrupt:
            print("Stopping...")
            break



def start():
    if len(sys.argv) < 3:
        print("Declare iface and model savefile to run IDS")
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = ModelPipeline.load(sys.argv[2], device)
    threshold = 1 - pipeline.best_threshold
    print("Threshold is: ", threshold)

    thread = threading.Thread(target=flow_reader, args=(pipeline.streamer,sys.argv[1]), daemon=True)
    thread.start()
    main_loop(pipeline, threshold, device)


if __name__ == "__main__":
    start()
