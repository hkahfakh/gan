# coding:utf-8
import numpy as np
from sklearn import preprocessing


# 直接标准化不知道怎么处理分母为0  所以直接调函数了
def standardization(data):
    X_scale = preprocessing.scale(data)
    return X_scale


# 数据的处理  生成测试集和训练集
def data_split(X, y, rate=0.9):
    splitIndex = int(X.shape[0] * rate)
    train_X, train_y = X[:splitIndex], y[:splitIndex]
    test_X, test_y = X[splitIndex:], y[splitIndex:]

    return train_X, train_y, test_X, test_y


def data_UNSW_txt(saveFlag=0):
    f = open("./dataSet/UNSW-NB15_1.txt", encoding='utf-8')
    lines = f.readlines()
    f.close()
    lines[0] = lines[0][1:]  # 去掉了一个多余的符号
    data = list()

    # 在这先不对他们进行处理    只是txt转npy
    for line in lines:
        info = line.split(",")
        info[-1] = info[-1][:-1]
        data.append(info)

    a = np.array(data)
    if saveFlag == 1:
        np.save("./dataSet/UNSW_rawText.npy", a)  # 保存为.npy格式
    return a


# 因为最后是需要检测数据是否为恶意   并不需要确定是哪种攻击
def data_UNSW_npy(data, saveFlag=0):
    # 前四个感觉不需要保存了   因为ip和端口好像没意义
    proto = ['3pc', 'a/n', 'aes-sp3-d', 'any', 'argus', 'aris', 'arp', 'ax.25', 'bbn-rcc', 'bna', 'br-sat-mon', 'cbt',
             'cftp', 'chaos', 'compaq-peer', 'cphb', 'cpnx', 'crtp', 'crudp', 'dcn', 'ddp', 'ddx', 'dgp', 'egp',
             'eigrp', 'emcon', 'encap', 'esp', 'etherip', 'fc', 'fire', 'ggp', 'gmtp', 'gre', 'hmp', 'i-nlsp', 'iatp',
             'ib', 'icmp', 'idpr', 'idpr-cmtp', 'idrp', 'ifmp', 'igmp', 'igp', 'il', 'ip', 'ipcomp',
             'ipcv', 'ipip', 'iplt', 'ipnip', 'ippc', 'ipv6', 'ipv6-frag', 'ipv6-no', 'ipv6-opts', 'ipv6-route',
             'ipx-n-ip', 'irtp', 'isis', 'iso-ip', 'iso-tp4', 'kryptolan', 'l2tp', 'larp', 'leaf-1', 'leaf-2',
             'merit-inp', 'mfe-nsp', 'mhrp', 'micp', 'mobile', 'mtp', 'mux', 'narp', 'netblt', 'nsfnet-igp', 'nvp',
             'ospf', 'pgm', 'pim', 'pipe', 'pnni', 'pri-enc', 'prm', 'ptp', 'pup', 'pvp', 'qnx', 'rdp',
             'rsvp', 'rtp', 'rvd', 'sat-expak', 'sat-mon', 'sccopmce', 'scps', 'sctp', 'sdrp',
             'secure-vmtp', 'sep', 'skip', 'sm', 'smp', 'snp', 'sprite-rpc', 'sps', 'srp', 'st2', 'stp', 'sun-nd',
             'swipe', 'tcf', 'tcp', 'tlsp', 'tp++', 'trunk-1', 'trunk-2', 'ttp', 'udp', 'udt', 'unas', 'uti', 'vines',
             'visa', 'vmtp', 'vrrp', 'wb-expak', 'wb-mon', 'wsn', 'xnet', 'xns-idp', 'xtp', 'zero']

    state = ['ACC', 'CLO', 'CON', 'ECO', 'ECR', 'FIN', 'INT', 'MAS', 'PAR', 'REQ', 'RST', 'TST', 'TXD', 'URH', 'URN',
             'no']
    # 没有就是良性
    attack_cat = ['', ' Fuzzers', 'Analysis', 'Backdoors', 'DoS', 'Exploits', 'Generic',
                  'Reconnaissance', 'Shellcode', 'Worms']
    # -是没有协议使用
    service = ['-', 'dhcp', 'dns', 'ftp', 'ftp-data', 'http', 'irc', 'pop3', 'radius', 'smtp',
               'snmp', 'ssh', 'ssl']
    # 连续变量为1
    continuous = [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1,
                  0, 1, 1, 1, 1, 1, 1, 1, 1]
    data = data[:, 5:]
    # 非数字字符串转为数字
    for i in range(data.shape[0]):
        data[:, 0][i] = state.index(data[:, 0][i])
        data[:, 8][i] = service.index(data[:, 8][i])
        data[:, 42][i] = attack_cat.index(data[:, 42][i])

    data = np.delete(data, -2, axis=1)  # 因为只需要判断数据是否为恶意所以不需要具体攻击标签
    data = data.astype(np.float64)

    for j in range(len(continuous)):
        if continuous[j] == 1:
            data[:, j] = standardization(data[:, j])
    if saveFlag == 1:
        np.save("./dataSet/UNSW_finally.npy", data)
    return data


def data_KDD_txt(saveFlag=0):
    f = open(".\dataSet\KDDTrain+_20Percent.txt")
    lines = f.readlines()
    f.close()
    data = list()

    # 在这先不对他们进行处理    只是txt转npy
    for line in lines:
        info = line.split(",")
        info[-1] = info[-1][:-1]
        data.append(info)

    a = np.array(data)
    if saveFlag == 1:
        np.save("./dataSet/KDD_rawText.npy", a)  # 保存为.npy格式
    return a


def data_KDD_npy(data, saveFlag=0):
    # 属性   1  2  3
    protocol_type = ['tcp', 'icmp', 'udp']
    service = ['IRC', 'X11', 'Z39_50', 'auth', 'bgp', 'courier', 'csnet_ns', 'ctf', 'daytime', 'discard', 'domain',
               'domain_u', 'echo', 'eco_i', 'ecr_i', 'efs', 'exec', 'finger', 'ftp', 'ftp_data', 'gopher', 'hostnames',
               'http', 'http_443', 'http_8001', 'imap4', 'iso_tsap', 'klogin', 'kshell', 'ldap', 'link', 'login', 'mtp',
               'name', 'netbios_dgm',
               'netbios_ns', 'netbios_ssn', 'netstat', 'nnsp', 'nntp', 'ntp_u', 'other', 'pm_dump', 'pop_2', 'pop_3',
               'printer', 'private', 'red_i', 'remote_job', 'rje',
               'shell', 'smtp', 'sql_net', 'ssh', 'sunrpc', 'supdup', 'systat', 'telnet', 'tim_i', 'time', 'urh_i',
               'urp_i', 'uucp', 'uucp_path', 'vmnet', 'whois']
    flag = ['OTH', 'REJ', 'RSTO', 'RSTOS0', 'RSTR', 'S0', 'S1', 'S2', 'S3', 'SF', 'SH']
    # 标签
    dst_host_srv_rerror_rate = ['normal', 'back', 'buffer_overflow', 'ftp_write', 'guess_passwd', 'imap', 'ipsweep',
                                'land',
                                'loadmodule', 'multihop', 'neptune', 'nmap', 'phf', 'pod', 'portsweep',
                                'rootkit', 'satan',
                                'smurf', 'spy', 'teardrop', 'warezclient', 'warezmaster']

    # 连续变量为1   第一个照着论文最后打的   论文上表好像不对
    continuous = [1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  1, 1, 1, 1, 1, 1, 1, 0]
    continuous = [1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  1, 1, 1, 1, 1, 1, 1, 0]
    data = data[:, :-1]
    # 非数字字符串转为数字
    for i in range(data.shape[0]):
        data[:, 1][i] = protocol_type.index(data[:, 1][i])
        data[:, 2][i] = service.index(data[:, 2][i])
        data[:, 3][i] = flag.index(data[:, 3][i])
        # 恶意数据 标记为1
        if dst_host_srv_rerror_rate.index(data[:, -1][i]) == 0:
            data[:, -1][i] = 0
        else:
            data[:, -1][i] = 1

    data = data.astype(np.float64)

    for j in range(len(continuous)):
        if continuous[j] == 1:
            data[:, j] = standardization(data[:, j])
    if saveFlag == 1:
        np.save("./dataSet/KDD_finally.npy", data)
    return data


# 显示每列都有哪些元素    列元素去重
def value_category(data):
    for x in range(data.shape[1]):
        print(np.unique(data[:, x]))


def gen_KDD_npy():
    data = data_KDD_txt()
    data_KDD_npy(data, saveFlag=1)


def gen_UNSW_npy():
    data = data_UNSW_txt()
    data_UNSW_npy(data, saveFlag=1)


def get_data(path):
    a = np.load(path)
    # a = a.tolist()
    return a


if __name__ == '__main__':
    gen_KDD_npy()
