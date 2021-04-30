import pandas as pd

def data_preprocess(init_df):
    df = init_df.drop(columns=['ip.tos', 'tcp.options.mss_val'])
    final_df = df.dropna()
    return final_df

def add_label(df, ip_label_dict):
    # create list of known IP addresses
    known_ip_list = [key for key in ip_label_dict.keys()]
    
    # filter df based on known IP addresses
    all_ip_list = list(set(df['ip.src'].tolist()))
    unnecessary_ip_list = [item for item in all_ip_list if item not in known_ip_list] 
    filtered_df = df.copy()
    for ip in unnecessary_ip_list:
        filtered_df = filtered_df[filtered_df['ip.src'] != ip] 
    
    # add label to the filtered df
    for ip in known_ip_list:
        filtered_df.loc[filtered_df['ip.src'] == ip, "os"] = ip_label_dict[ip]
    return filtered_df
