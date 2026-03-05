from nfstream import NFPlugin
from collections import namedtuple

INIT = 0
FIN_0 = 1
FIN_1 = 2
TERM = 3

PacketData = namedtuple('PacketData', ['timestamp', 'size', 'ip_protocol', 'direction', 'tcp_flags'])

class PacketCollector(NFPlugin):

    def monitor_tcp_hanshake(self, flow, packet):
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
                    flow.udps.last_fin = packet.direction
            elif packet.rst:
                flow.udps.rst = True
            elif packet.ack:
                if flow.udps.status == FIN_1 and packet.direction != flow.udps.last_fin:
                    flow.udps.status = TERM
            
            if flow.udps.status == TERM:
                flow.expiration_id = -1

    def on_init(self, packet, flow):
        flow.udps.status = INIT
        flow.udps.fin_fwd = 0
        flow.udps.fin_bwd = 0
        flow.udps.last_fin = None
        flow.udps.rst = False
        flow.udps.packets = []
        self._collect(packet, flow)
        self.monitor_tcp_hanshake(flow, packet)

    def on_update(self, packet, flow):
        self._collect(packet, flow)
        self.monitor_tcp_hanshake(flow, packet)


    def _collect(self, packet, flow):
        flow.udps.packets.append(PacketData(
            timestamp=packet.time / 10 ** 6,
            size=packet.payload_size,
            ip_protocol=packet.protocol,
            direction=packet.direction,
            tcp_flags=self._tcp_flags(packet)
        ))

    def _tcp_flags(self, packet):
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


