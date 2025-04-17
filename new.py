import bencode

with open("10G_data_new.torrent",'rb') as f:
    data = f.read()
    data = bencode.decode(data)

print("aa")