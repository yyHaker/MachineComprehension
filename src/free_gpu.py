import sys
import argparse
from pexpect import pxssh


def send_command(s, cmd):
    s.sendline(cmd)
    s.prompt()
    return str(s.before).replace('\\r\\n', '\r\n')


def connect(host, user, password):
    try:
        s = pxssh.pxssh()
        s.login(host, user, password)
        return s
    except:
        print('login error with host:', host)
        exit(0)


def get_free_info(nvidia_info):
    lines = nvidia_info.split('\r\n')[1:-1]
    # 找到Processes所在行
    for idx, line in enumerate(lines):
        if 'Processes' in line:
            processs_idx = idx
    # 计算可用GPU个数
    gpu_count = int((processs_idx - 10) / 3)
    # 计算已占用的GPU ID
    used_gpus = set()
    for line in lines[processs_idx+3:-2]:
        used_gpus.add(int(line[5]))
    # 得到未使用的GPU信息
    free_gpu_idx = []
    for i in range(gpu_count):
        if i not in used_gpus:
            free_gpu_idx.append(i)
    free_info = ''
    if free_gpu_idx:
        for idx in free_gpu_idx:
            free_info += '\n'.join(lines[7+idx*3:7+(idx+1)*3-1])
            free_info += '\n'
    return free_info


def main():
    cmd = argparse.ArgumentParser(description='Check free gpus')
    cmd.add_argument('--user', default='yle', type=str, help='user name.')
    cmd.add_argument('--password', default='leyuan', type=str, help='password.')
    cmd.add_argument('--gpus', default='05,06,08,09,10,11,12,13,14,15', type=str, help='gpus to check')
    args = cmd.parse_args()
    node_list = ['gpu' + gpu for gpu in args.gpus.split(',')]
    print('Check free GPUs on :', ','.join(node_list))
    for node in node_list:
        s = connect(node, args.user, args.password)
        nvidia_info = send_command(s, 'nvidia-smi')
        free_info = get_free_info(nvidia_info)
        if free_info:
            print('Free GPUs on '+node)
            print(free_info)
        else:
            print('No free GPU on '+node)


if __name__ == '__main__':
    main()