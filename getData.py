# coding:utf-8
import numpy as np
import os

def pre_processing():
    f = open(".\kddcup.data_10_percent_corrected")
    lines = f.readlines()
    data = list()
    protocol = ['tcp', 'icmp', 'udp']
    service = ['discard', 'IRC', 'rje', 'nnsp', 'klogin', 'netbios_ssn', 'iso_tsap', 'ldap', 'mtp', 'finger', 'courier',
               'link',
               'echo', 'time', 'remote_job', 'telnet', 'eco_i', 'tim_i', 'ecr_i', 'other', 'Z39_50', 'hostnames',
               'csnet_ns',
               'kshell', 'uucp_path', 'domain', 'nntp', 'uucp', 'bgp', 'pop_3', 'urp_i', 'auth', 'urh_i', 'efs',
               'daytime',
               'sunrpc', 'pm_dump', 'http', 'shell', 'http_443', 'systat', 'name', 'red_i', 'ntp_u', 'ftp_data', 'ftp',
               'ssh',
               'domain_u', 'netbios_ns', 'ctf', 'netbios_dgm', 'sql_net', 'printer', 'netstat', 'tftp_u', 'gopher',
               'whois',
               'imap4', 'login', 'supdup', 'smtp', 'pop_2', 'vmnet', 'private', 'X11', 'exec']
    dst_host_srv_rerror_rate = ['ipsweep', 'multihop', 'satan', 'ftp_write', 'guess_passwd', 'phf', 'spy', 'imap',
                                'buffer_overflow', 'rootkit',
                                'perl', 'normal', 'nmap', 'loadmodule', 'smurf', 'neptune', 'pod', 'portsweep', 'back',
                                'land', 'warezmaster',
                                'teardrop', 'warezclient']
    flag = ['OTH', 'S0', 'S2', 'REJ', 'RSTR', 'SF', 'RSTOS0', 'S3', 'SH', 'RSTO', 'S1']

    for line in lines:
        info = line.split(",")
        # 这是将字符串value替换成数字value
        info[1] = protocol.index(info[1])
        info[2] = service.index(info[2])
        info[3] = flag.index(info[3])
        info[-1] = info[-1][:-2]
        info[-1] = dst_host_srv_rerror_rate.index(info[-1])
        #info = [int(float(x)) for x in info]

        data.append(info)
    f.close()

    a = np.array(data)
    np.save("dataSet/a.npy", a)  # 保存为.npy格式


def get_NSLKDD_data():
    a = np.load("./networkData_number.npy")
    #a = a.tolist()
    return a

def get_NSLKDD_data():
    a = np.load("./networkData_number.npy")
    #a = a.tolist()
    return a

def load_NSLKDD():
    print(os.path.abspath(os.path.join(os.getcwd(), "../..")))
    f = open(os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/dataset/KDDTrain+_20Percent.txt")
    lines = f.readlines()
    data = list()
    protocol = ['tcp', 'icmp', 'udp']
    service = ['discard', 'IRC', 'rje', 'nnsp', 'klogin', 'netbios_ssn', 'iso_tsap', 'ldap', 'mtp', 'finger', 'courier',
               'link',
               'echo', 'time', 'remote_job', 'telnet', 'eco_i', 'tim_i', 'ecr_i', 'other', 'Z39_50', 'hostnames',
               'csnet_ns',
               'kshell', 'uucp_path', 'domain', 'nntp', 'uucp', 'bgp', 'pop_3', 'urp_i', 'auth', 'urh_i', 'efs',
               'daytime',
               'sunrpc', 'pm_dump', 'http', 'shell', 'http_443', 'systat', 'name', 'red_i', 'ntp_u', 'ftp_data', 'ftp',
               'ssh',
               'domain_u', 'netbios_ns', 'ctf', 'netbios_dgm', 'sql_net', 'printer', 'netstat', 'tftp_u', 'gopher',
               'whois',
               'imap4', 'login', 'supdup', 'smtp', 'pop_2', 'vmnet', 'private', 'X11', 'exec']
    dst_host_srv_rerror_rate = ['ipsweep', 'multihop', 'satan', 'ftp_write', 'guess_passwd', 'phf', 'spy', 'imap',
                                'buffer_overflow', 'rootkit',
                                'perl', 'normal', 'nmap', 'loadmodule', 'smurf', 'neptune', 'pod', 'portsweep', 'back',
                                'land', 'warezmaster',
                                'teardrop', 'warezclient']
    flag = ['OTH', 'S0', 'S2', 'REJ', 'RSTR', 'SF', 'RSTOS0', 'S3', 'SH', 'RSTO', 'S1']

    for line in lines:
        info = line.split(",")

        # # 这是将字符串value替换成数字value
        # info[1] = protocol.index(info[1])
        # info[2] = service.index(info[2])
        # info[3] = flag.index(info[3])
        # info[-1] = info[-1][:-2]
        # info[-1] = dst_host_srv_rerror_rate.index(info[-1])
        # # info = [int(float(x)) for x in info]

        data.append(info)
    f.close()

    a = np.array(data)
    np.save("NSLKDD.npy", a)  # 保存为.npy格式

def load_UNSWNB15():
    f = open(os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/dataset/UNSW-NB15_1.txt")
    lines = f.readlines()
    data = list()
    protocol = ['tcp', 'icmp', 'udp']
    service = ['discard', 'IRC', 'rje', 'nnsp', 'klogin', 'netbios_ssn', 'iso_tsap', 'ldap', 'mtp', 'finger', 'courier',
               'link',
               'echo', 'time', 'remote_job', 'telnet', 'eco_i', 'tim_i', 'ecr_i', 'other', 'Z39_50', 'hostnames',
               'csnet_ns',
               'kshell', 'uucp_path', 'domain', 'nntp', 'uucp', 'bgp', 'pop_3', 'urp_i', 'auth', 'urh_i', 'efs',
               'daytime',
               'sunrpc', 'pm_dump', 'http', 'shell', 'http_443', 'systat', 'name', 'red_i', 'ntp_u', 'ftp_data', 'ftp',
               'ssh',
               'domain_u', 'netbios_ns', 'ctf', 'netbios_dgm', 'sql_net', 'printer', 'netstat', 'tftp_u', 'gopher',
               'whois',
               'imap4', 'login', 'supdup', 'smtp', 'pop_2', 'vmnet', 'private', 'X11', 'exec']
    dst_host_srv_rerror_rate = ['ipsweep', 'multihop', 'satan', 'ftp_write', 'guess_passwd', 'phf', 'spy', 'imap',
                                'buffer_overflow', 'rootkit',
                                'perl', 'normal', 'nmap', 'loadmodule', 'smurf', 'neptune', 'pod', 'portsweep', 'back',
                                'land', 'warezmaster',
                                'teardrop', 'warezclient']
    flag = ['OTH', 'S0', 'S2', 'REJ', 'RSTR', 'SF', 'RSTOS0', 'S3', 'SH', 'RSTO', 'S1']

    for line in lines:
        info = line.split(",")

        # # 这是将字符串value替换成数字value
        # info[1] = protocol.index(info[1])
        # info[2] = service.index(info[2])
        # info[3] = flag.index(info[3])
        # info[-1] = info[-1][:-2]
        # info[-1] = dst_host_srv_rerror_rate.index(info[-1])
        # # info = [int(float(x)) for x in info]

        data.append(info)
    f.close()

    a = np.array(data)
    np.save("UNSWNB15.npy", a)  # 保存为.npy格式

'''
 {'tcp', 'icmp', 'udp'}, 
 {'discard', 'IRC', 'rje', 'nnsp', 'klogin', 'netbios_ssn', 'iso_tsap', 'ldap', 'mtp', 'finger', 'courier', 'link', 'echo', 'time', 'remote_job', 'telnet', 'eco_i', 'tim_i', 'ecr_i', 'other', 'Z39_50', 'hostnames', 'csnet_ns', 'kshell', 'uucp_path', 'domain', 'nntp', 'uucp', 'bgp', 'pop_3', 'urp_i', 'auth', 'urh_i', 'efs', 'daytime', 'sunrpc', 'pm_dump', 'http', 'shell', 'http_443', 'systat', 'name', 'red_i', 'ntp_u', 'ftp_data', 'ftp', 'ssh', 'domain_u', 'netbios_ns', 'ctf', 'netbios_dgm', 'sql_net', 'printer', 'netstat', 'tftp_u', 'gopher', 'whois', 'imap4', 'login', 'supdup', 'smtp', 'pop_2', 'vmnet', 'private', 'X11', 'exec'}, 
 {'OTH', 'S0', 'S2', 'REJ', 'RSTR', 'SF', 'RSTOS0', 'S3', 'SH', 'RSTO', 'S1'}
 {'ipsweep', 'multihop', 'satan', 'ftp_write', 'guess_passwd', 'phf', 'spy', 'imap', 'buffer_overflow', 'rootkit', 'perl', 'normal', 'nmap', 'loadmodule', 'smurf', 'neptune', 'pod', 'portsweep', 'back', 'land', 'warezmaster', 'teardrop', 'warezclient'}
'''
if __name__ == '__main__':
    load_UNSWNB15()
