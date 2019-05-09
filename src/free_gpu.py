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
        return None


def get_free_gpu(nvidia_info, min_free_mem=-1):
    '''
    if min_free_mem==-1:
        找到未被占用的节点信息
    else：
        找到剩余显存不小于min_free_mem的节点并返回节点信息
    '''
    lines = nvidia_info.split('\r\n')

    # 找到Processes所在行
    for idx, line in enumerate(lines):
        if 'Processes' in line:
            processs_idx = idx
            break
    # 计算可用GPU个数
    gpu_count = int((processs_idx - 10) / 3)
    if min_free_mem != -1:
        # 找到剩余显存不小于min_free_mem的节点
        free_gpu_idx = []
        for i in range(gpu_count):
            line = lines[9+i*3]
            mem_info = line.replace(' ', '').split('|')[2]
            used_mem, all_mem = mem_info.replace('MiB', '').split('/')
            if int(all_mem) - int(used_mem) >= min_free_mem:
                free_gpu_idx.append(i)
    else:
        # 计算已占用的GPU ID
        used_gpus = set()
        for line in lines[processs_idx+3:-2]:
            if line[3:5] != 'No':
                used_gpus.add(int(line[5]))
        # 得到未使用的GPU信息
        free_gpu_idx = []
        for i in range(gpu_count):
            if i not in used_gpus:
                free_gpu_idx.append(i)
    # 得到符合条件的节点信息
    free_info = ''
    for idx in free_gpu_idx:
        free_info += '\n'.join(lines[8+idx*3:8+(idx+1)*3-1])
        free_info += '\n'
    return free_info


def main():
    cmd = argparse.ArgumentParser(description='Check free gpus')
    cmd.add_argument('--user', '-u', default='yle', type=str, help='user name.')
    cmd.add_argument('--password', '-p', default='leyuan', type=str, help='password.')
    cmd.add_argument('--gpus', '-g', default='05,06,07,08,09,10,11,12,13,14,15,16', type=str, help='gpus to check')
    cmd.add_argument('--min_free_mem', '-m', default=-1, type=int, help='Min free memory')
    args = cmd.parse_args()
    node_list = ['gpu' + gpu for gpu in args.gpus.split(',')]
    print('Check free GPUs on :', ','.join(node_list))
    for node in node_list:
        s = connect(node, args.user, args.password)
        if s:
            nvidia_info = send_command(s, 'nvidia-smi')
            free_info = get_free_gpu(nvidia_info, args.min_free_mem)
            if free_info:
                print('Free GPUs on '+node)
                print(free_info)
                print('\n')
            else:
                print('No free GPU on '+node)
                print('\n')


if __name__ == '__main__':
    main()
