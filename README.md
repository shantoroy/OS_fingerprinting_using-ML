# Operating System Fingerprinting using ML Classifiers
The project is about fingerprinting operating systems using different multi-class
classification algorithms. I tried to look into required features for OS fingerprinting
and find accuracy of different classifiers based on different labeling (base OS platform, 
e.g., Windows vs OS versions, e.g., Win 7,8,10). The accuracy is almost 100% if labeled 
as base platform only (Windows, Ubuntu, macOS). However, the accuray is lower when 
labeled as the OS version (see below in the Ground Truth section).

## Breaking Down PCAP files (MB)
To test with a small file, a large pcap file can be broken down to a small file 
using the following command
```
$ tcpdump -r old_file.pcap -w new_file.pcap -C 100
```

## Convert PCAP to CSV
There are many ways to convert a **PCAP** file to a **CSV** file.
The best way to do that is using **tshark** which is the command-line version of Wireshark
In an Ubuntu machine, it can be installed using the following command
```
$ sudo apt install tshark
```

Now, convert the input pcap file to a CSV file as follows:
```
$ tshark -r input.pcap -T fields -E header=y -E separator=, -E quote=d -E occurrence=f \
-e ip.version -e ip.hdr_len -e ip.tos -e ip.id -e ip.flags -e ip.flags.rb -e ip.flags.df \ 
-e ip.flags.mf -e ip.frag_offset -e ip.ttl -e ip.proto -e ip.checksum -e ip.src -e ip.dst \ 
-e ip.len -e ip.dsfield -e tcp.srcport -e tcp.dstport -e tcp.seq -e tcp.ack -e tcp.len \ 
-e tcp.hdr_len -e tcp.flags -e tcp.flags.fin -e tcp.flags.syn -e tcp.flags.reset \ 
-e tcp.flags.push -e tcp.flags.ack -e tcp.flags.urg -e tcp.flags.cwr -e tcp.window_size \ 
-e tcp.checksum -e tcp.urgent_pointer -e tcp.options.mss_val > output.csv
```

Note that, there are many other options I did not include in the CSV file. If more interested, 
you can check other options as well. For example, [this article](https://www.linkedin.com/pulse/build-machine-learning-model-network-flow-tao-liu/) did the following for a different purpose.
```
$ tshark -r $1 -T fields -E header=y -E separator=, -E quote=d -E occurrence=f -e ip.src -e ip.dst -e ip.len -e ip.flags.df -e ip.flags.mf \
-e ip.fragment -e ip.fragment.count -e ip.fragments -e ip.ttl -e ip.proto -e tcp.window_size -e tcp.ack -e tcp.seq -e tcp.len -e tcp.stream -e tcp.urgent_pointer \
-e tcp.flags -e tcp.analysis.ack_rtt -e tcp.segments -e tcp.reassembled.length -e ssl.handshake -e ssl.record -e ssl.record.content_type -e ssl.handshake.cert_url.url_len \
-e ssl.handshake.certificate_length -e ssl.handshake.cert_type -e ssl.handshake.cert_type.type -e ssl.handshake.cert_type.types -e ssl.handshake.cert_type.types_len \
-e ssl.handshake.cert_types -e ssl.handshake.cert_types_count -e dtls.handshake.extension.len -e dtls.handshake.extension.type -e dtls.handshake.session_id \
-e dtls.handshake.session_id_length -e dtls.handshake.session_ticket_length -e dtls.handshake.sig_hash_alg_len -e dtls.handshake.sig_len -e dtls.handshake.version \
-e dtls.heartbeat_message.padding -e dtls.heartbeat_message.payload_length -e dtls.heartbeat_message.payload_length.invalid -e dtls.record.content_type -e dtls.record.content_type \
-e dtls.record.length -e dtls.record.sequence_number -e dtls.record.version -e dtls.change_cipher_spec -e dtls.fragment.count -e dtls.handshake.cert_type.types_len \
-e dtls.handshake.certificate_length -e dtls.handshake.certificates_length -e dtls.handshake.cipher_suites_length -e dtls.handshake.comp_methods_length -e dtls.handshake.exponent_len \
-e dtls.handshake.extension.len -e dtls.handshake.extensions_alpn_str -e dtls.handshake.extensions_alpn_str_len -e dtls.handshake.extensions_key_share_client_length \
-e http.request -e udp.port -e frame.time_relative -e frame.time_delta -e tcp.time_relative -e tcp.time_delta > $2
```

## Run the File
Four options for the `main.py` file
* -file     : the input CSV file
* -features : the column index of selected features in the new generated CSV file (`labeled_dataset.csv`)
* -label    : the label index of the new generated CSV file (`labeled_dataset.csv`)
* -test     : test size for dividing the data into train and test sizes

Now, the program can be run using the following command
```sh
$ python main.py -file thursday-100M-v2.csv -features 1,5,8,13,17,18,19,20,22,23,24,25,26,29 -label 32 -test 0.2 > report.txt
```

## Classification Methods
The following `multi-class classification` algorithms have been used to train and test the dataset 
    * Logistic Regression Classifier
    * K-Neighbor Classifier
    * SVM (Linear) Classifier
    * SVM (RBF) Classifier
    * Naive Bayes Classifier
    * Decision Tree Classifier
    * Random Forest Classifier

## Feature Selection Algorithms
Features are selected based on the following feature ranking algorithms
* Univariate Selection (ANOVA f-val, Chi-Squared)
* Recursive Feature Elimination 
* Extra Tree Classifier for feature importance


## Considered Features
* ip.hdr_len
* ip.flags.df
* ip.ttl
* ip.len
* tcp.seq
* tcp.ack
* tcp.len
* tcp.hdr_len
* tcp.flags.fin
* tcp.flags.syn
* tcp.flags.reset
* tcp.flags.push
* tcp.flags.ack
* tcp.window_size


## Used Dataset
A very small part of the [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html)

## Labeling Data (Ground Truth)
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

## Future Work
* Tune the Classification Hyperparameters
* Use DNN (MLP and LSTM) to classify
* Use large Dataset
* Improve Code Documentation


## Help
* [ML Classification Template](https://gist.github.com/shantoroy/2172937f5157998069d667b362e3fe81)
* [Feature Selection Template](https://gist.github.com/shantoroy/9bb4da0b2a281e3c91cc836045b6c74d)
* [Feature Selection ML](https://machinelearningmastery.com/feature-selection-machine-learning-python/)