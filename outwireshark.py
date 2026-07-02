# 📌 GEREKLİ KÜTÜPHANELER
# pyshark → Wireshark pcap dosyasını okumamızı sağlar
# pandas → veriyi tabloya (CSV) çevirmek için

import pyshark
import pandas as pd

# 📌 PCAP DOSYASINI AÇIYORUZ
# Buraya Wireshark ile kaydettiğin .pcap dosyasını koy
cap = pyshark.FileCapture(r"C:\Users\gamze\OneDrive\Belgeler\trafik.pcapng")

# 📌 Boş liste oluşturuyoruz
# Her paketi buraya satır satır ekleyeceğiz
data = []

# 📌 PCAP içindeki her paketi tek tek geziyoruz
for packet in cap:
    try:
        # 📌 IP ve TCP/UDP bilgisi varsa alıyoruz
        data.append({
            "time": packet.sniff_time,        # paketin zamanı
            "src_ip": packet.ip.src,          # gönderen IP
            "dst_ip": packet.ip.dst,          # alan IP
            "protocol": packet.transport_layer,  # TCP / UDP
            "length": packet.length           # paket boyutu
        })

    except:
        # 📌 bazı paketlerde IP yok (ARP vs.), onları atlıyoruz
        pass

# 📌 Listeyi tabloya çeviriyoruz
df = pd.DataFrame(data)

# 📌 CSV dosyası olarak kaydediyoruz
df.to_csv("packets.csv", index=False)

print("✔ Paketler CSV olarak kaydedildi: packets.csv")