import json
from datetime import datetime, timedelta

DATE_FORMAT_INTERNAL = '%d/%m/%Y %I:%M:%S %p'

def load_attack_config(filepath):
    with open(filepath, 'r') as f:
        full_config = json.load(f)
    
    timezone_offset_hours = full_config.get('config', {}).get('timezone_offset_hours', 0)
    raw_attacks = full_config.get('attacks', {})
    
    for attack_name, attack_data in raw_attacks.items():
        attack_data['start_time'] = datetime.strptime(attack_data['start_time'], DATE_FORMAT_INTERNAL)
        attack_data['end_time'] = datetime.strptime(attack_data['end_time'], DATE_FORMAT_INTERNAL)
        
    return timezone_offset_hours, raw_attacks

def is_malicious(flow, timezone_offset_hours, attacks_config):
    flow_start_utc = datetime.utcfromtimestamp(flow.bidirectional_first_seen_ms / 1000000.0)
    flow_start = flow_start_utc - timedelta(hours=timezone_offset_hours)

    for attack in attacks_config.values():
        t_start = attack['start_time']
        t_end = attack['end_time']
        attacker = attack.get('attacker', [])
        victim = attack.get('victim', [])

        if not (t_start <= flow_start <= t_end):
            continue 

        ip_match = False
        if "cond" in attack and (attack["cond"] == flow.src_ip or attack["cond"] == flow.dst_ip):
            ip_match = True
        elif (flow.src_ip in attacker) and (flow.dst_ip in victim):
            ip_match = True
        
        if ip_match:
            port_match = True
            
            if 'dst_port' in attack and attack['dst_port'] != flow.dst_port:
                port_match = False
            if 'src_port' in attack and attack['src_port'] != flow.src_port:
                port_match = False
                
            if port_match:
                return attack['label']

    return 0
