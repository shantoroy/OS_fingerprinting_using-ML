# Operating System Fingerprinting using ML Classifiers

**Details on the way...**

## Convert PCAP to CSV
```
tshark -r thursday.pcap -T fields -E header=y -E separator=, -E quote=d -E occurrence=f \
-e ip.version -e ip.hdr_len -e ip.tos -e ip.id -e ip.flags -e ip.flags.rb -e ip.flags.df \ 
-e ip.flags.mf -e ip.frag_offset -e ip.ttl -e ip.proto -e ip.checksum -e ip.src -e ip.dst \ 
-e ip.len -e ip.dsfield -e tcp.srcport -e tcp.dstport -e tcp.seq -e tcp.ack -e tcp.len \ 
-e tcp.hdr_len -e tcp.flags -e tcp.flags.fin -e tcp.flags.syn -e tcp.flags.reset \ 
-e tcp.flags.push -e tcp.flags.ack -e tcp.flags.urg -e tcp.flags.cwr -e tcp.window_size \ 
-e tcp.checksum -e tcp.urgent_pointer -e tcp.options.mss_val > thursday.csv
```

## Run the File
```sh
$ python main.py -file thursday-100M-v2.csv -features 1,5,8,13,17,18,19,20,22,23,24,25,26,29 -label 32 -test 0.2
```

## Classification Methods
    * Logistic Regression Classifier
    * K-Neighbor Classifier
    * SVM (Linear) Classifier
    * SVM (RBF) Classifier
    * Naive Bayes Classifier
    * Decision Tree Classifier
    * Random Forest Classifier

## Considered Features
    * ip.ttl
    * ip.flags
    * ip.flags.df
    * ip.flags.mf
    * ip.flags.rb
    * tcp.hdr\_len
    * tcp.flags.fin
    * tcp.flags.syn
    * tcp.flags.reset
    * tcp.flags.push
    * tcp.flags.ack
    * tcp.flags.urg
    * tcp.window\_size

## Used Dataset
A very small part of the [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html)

## Ground Truth
```
ip_dict = {
    '192.168.10.51': 'Ubuntu server 12',
    '192.168.10.19': 'Ubuntu 14.4',
    '192.168.10.17': 'Ubuntu 14.4',
    '192.168.10.16': 'Ubuntu 16.4',
    '192.168.10.12': 'Ubuntu 16.4',
     '192.168.10.9': 'Win 7',
     '192.168.10.5': 'Win 8.1',
     '192.168.10.8': 'Win Vista',
    '192.168.10.14': 'Win 10',
    '192.168.10.15': 'Win 10',
    '192.168.10.25': 'macOS'
}
```
