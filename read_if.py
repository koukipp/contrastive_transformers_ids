from nfstream import NFPlugin, NFStreamer
from model import BERT
import pandas as pd
import torch
import queue
import threading
import time

INIT = 0
FIN_0 = 1
FIN_1 = 2
TERM = 3


# --- Configuration ---
BATCH_SIZE = 128
TIMEOUT = 2.0  # seconds
INTERFACE = "vf0_0"  # change to your actual network interface


# --- Shared Queue for Flow Tensors ---
flow_queue = queue.Queue()



def flow_to_tensor(flow, max_packets=32):
    packet_sequence = flow.udps.packet_sequence[:max_packets]  # truncate
    padded_packets = []
    mask = []

    prev_time = None
    for pkt in packet_sequence:
        current_time = pkt["time"] / 1000000
        iat = (current_time - prev_time) if prev_time is not None else 0.0
        prev_time = current_time

        padded_packets.append([
            iat,
            pkt["direction"] + 3,
            pkt["ip_protocol"] + 3,
            pkt["tcp_flags"] + 3 if pkt["tcp_flags"] is not None else 3,
            pkt["size"] / 1400
        ])
        mask.append(1.0)
    
    # Pad if needed
    while len(padded_packets) < max_packets:
        padded_packets.append([0.0] * 5)
        mask.append(0.0)
    
    start_packet = [0.0, 1.0, 1.0, 1.0, 0.0]  # [iat, direction, size, protocol, flags]
    padded_packets.insert(0, start_packet)
    mask.insert(0, 1.0)

    flow_tensor =  torch.tensor(padded_packets, dtype=torch.float32)
    mask_tensor = ~torch.tensor(mask, dtype=torch.bool)

    return flow_tensor, mask_tensor

def get_tcp_flags(packet):
    flags = 0
    if packet.fin: flags |= 0x01
    if packet.syn: flags |= 0x02
    if packet.rst: flags |= 0x04
    if packet.psh: flags |= 0x08
    if packet.ack: flags |= 0x10
    if packet.urg: flags |= 0x20
    if packet.ece: flags |= 0x40
    if packet.cwr: flags |= 0x80
    return flags

def monitor_tcp_hanshake(flow, packet):
    if packet.protocol == 6:
        if packet.fin:
            if flow.udps.status == INIT:
                flow.udps.status = FIN_0
            if not packet.direction:
                flow.udps.fin_fwd += 1
            elif packet.direction:
                flow.udps.fin_bwd += 1
        
            if flow.udps.fin_fwd and flow.udps.fin_bwd:
                flow.udps.status = FIN_1
        elif packet.rst:
            flow.udps.rst = True
        elif packet.ack:
            if flow.udps.status == FIN_1:
                flow.udps.status = TERM
        
        if flow.udps.status == TERM:
            flow.expiration_id = -1

class PacketMetadataTracker(NFPlugin):
    def on_init(self, packet, flow):
        flow.udps.packet_sequence = []
        flow.udps.status = INIT
        flow.udps.fin_fwd = 0
        flow.udps.fin_bwd = 0
        flow.udps.rst = False
        packet_info = {
            "time": packet.time,
            "direction": packet.direction,
            "size": packet.payload_size,
            "ip_protocol": packet.protocol,
            "tcp_flags": get_tcp_flags(packet)
        }
        flow.udps.packet_sequence.append(packet_info)
        monitor_tcp_hanshake(flow, packet)

    def on_update(self, packet, flow):
        packet_info = {
            "time": packet.time,
            "direction": packet.direction,
            "size": packet.payload_size,
            "ip_protocol": packet.protocol,
            "tcp_flags": get_tcp_flags(packet)
        }

        flow.udps.packet_sequence.append(packet_info)

        monitor_tcp_hanshake(flow, packet)
        


# --- Background Thread: Continuously monitor and enqueue flow tensors ---
def flow_reader():
    streamer = NFStreamer(source="vf0_0",bpf_filter="ip",idle_timeout=120, active_timeout=120, udps=[PacketMetadataTracker()], decode_tunnels=True, accounting_mode=3,n_dissections=0)

    for flow in streamer:
        tensor, mask = flow_to_tensor(flow)
        flow_queue.put((flow, tensor, mask))


# --- Main Thread: Batch and process flows ---
def main_loop(model, y_pred, threshold):
    buffer_flows = []
    buffer_tensors = []
    buffer_masks = []
    last_batch_time = time.time()

    while True:
        try:
            now = time.time()
            # Wait at most TIMEOUT seconds for a new flow
            wait_time = max(0.0, TIMEOUT - (now - last_batch_time))
            try:
                flow, tensor, mask = flow_queue.get(timeout=wait_time)
                if not flow.act_timeout:
                    buffer_flows.append(flow)
                    buffer_tensors.append(tensor)
                    buffer_masks.append(mask)
            except queue.Empty:
                pass

            # Check batch conditions
            if len(buffer_tensors) >= BATCH_SIZE or (buffer_tensors and (time.time() - last_batch_time > TIMEOUT)):
                batched_flows = torch.stack(buffer_tensors)
                batched_masks = torch.stack(buffer_masks)

                with torch.no_grad():
                    embed = model.embeddings(batched_flows, batched_masks)
                    y_out = torch.nn.functional.normalize(embed, p=2, dim=1)
                    similarity = torch.mm(y_out, y_pred.T)
                    cos_sim, indices = similarity.max(dim=1)
                    flow_idx, = torch.where(cos_sim < threshold)
                    for idx in flow_idx.tolist():
                        flow = buffer_flows[idx]
                        print(f"Malicious flow: {cos_sim[idx]},{flow.src_ip},{flow.dst_ip},{flow.src_port},{flow.dst_port},{flow.bidirectional_first_seen_ms}")

                print(f"[Main Loop] Processed batch of {len(buffer_tensors)} flows")

                buffer_flows.clear()
                buffer_tensors.clear()
                buffer_masks.clear()
                last_batch_time = time.time()

        except KeyboardInterrupt:
            print("Stopping...")
            break


# --- Start Everything ---
def start():
    checkpoint = torch.load('bert_cont_nfs.pt', map_location=torch.device("cpu"))
    model = BERT(hidden_dim = 256)
    model.to("cpu")
    model.load_state_dict(checkpoint['model_state_dict'])
    y_pred = checkpoint['benign_arr'].to("cpu")
    threshold = 1 - checkpoint['threshold']
    print("Threshold is: ", threshold)
    model.eval()

    thread = threading.Thread(target=flow_reader, daemon=True)
    thread.start()
    main_loop(model, y_pred, threshold)


if __name__ == "__main__":
    start()
