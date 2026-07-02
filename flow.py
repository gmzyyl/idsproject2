# 📌 Wireshark pcap okuma kütüphanesi
import pyshark

# 📌 veri tablosu için pandas
import pandas as pd

# 📌 aynı flow'ları toplamak için sözlük
from collections import defaultdict

# 📌 PCAP dosyasını açıyoruz
cap = pyshark.FileCapture("example.pcap")

# 📌 Flow yapısı:
# key → (src_ip, src_port, dst_ip, dst_port)
# value → o flow'a ait paketler
flows = defaultdict(list)

# 📌 tüm paketleri geziyoruz
for packet in cap:
    try:
        # 📌 sadece TCP paketleri alıyoruz (basitlik için)
        if hasattr(packet, "ip") and hasattr(packet, "tcp"):

            # 📌 FLOW ANAHTARI
            key = (
                packet.ip.src,        # kaynak IP
                packet.tcp.srcport,   # kaynak port
                packet.ip.dst,        # hedef IP
                packet.tcp.dstport    # hedef port
            )

            # 📌 bu flow’a paket uzunluğunu ekle
            flows[key].append(int(packet.length))

    except:
        # 📌 bozuk / IP'siz paketleri geç
        pass

# 📌 sonuçları tabloya çevireceğiz
rows = []

# 📌 her flow'u tek satır yapıyoruz
for key, sizes in flows.items():

    rows.append({
        "src_ip": key[0],
        "src_port": key[1],
        "dst_ip": key[2],
        "dst_port": key[3],

        # 📌 CICFlowMeter tarzı özellikler
        "packet_count": len(sizes),               # kaç paket var
        "total_bytes": sum(sizes),                # toplam trafik
        "avg_packet_size": sum(sizes)/len(sizes)  # ortalama paket boyutu
    })

# 📌 dataframe oluştur
df = pd.DataFrame(rows)

# 📌 CSV kaydet
df.to_csv("flows.csv", index=False)

print("✔ Flow çıkarıldı: flows.csv")