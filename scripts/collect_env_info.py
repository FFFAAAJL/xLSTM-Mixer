#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Environment information collection script
Collects: OS, GPU model, CUDA version, PyTorch info, DataLoader multiprocessing info
"""
import sys
import platform
import subprocess

def get_os_info():
    """Get operating system information"""
    os_name = platform.system()
    os_version = platform.version()
    os_arch = platform.machine()
    return {
        'os_name': os_name,
        'os_version': os_version,
        'os_arch': os_arch,
        'platform': f"{os_name} {os_version} ({os_arch})"
    }

def get_gpu_info():
    """Get GPU information"""
    gpu_info = {}
    try:
        # Try using nvidia-smi
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            gpus = []
            for line in lines:
                if line.strip():
                    parts = line.split(',')
                    if len(parts) >= 2:
                        gpus.append({
                            'name': parts[0].strip(),
                            'driver_version': parts[1].strip()
                        })
            gpu_info['gpus'] = gpus
            gpu_info['gpu_count'] = len(gpus)
        else:
            gpu_info['error'] = 'nvidia-smi execution failed'
    except FileNotFoundError:
        gpu_info['error'] = 'nvidia-smi not found, may not have NVIDIA GPU or driver not installed'
    except subprocess.TimeoutExpired:
        gpu_info['error'] = 'nvidia-smi execution timeout'
    except Exception as e:
        gpu_info['error'] = f'Error getting GPU info: {str(e)}'
    
    return gpu_info

def get_cuda_info():
    """Get CUDA version information"""
    cuda_info = {}
    try:
        # Try using nvcc
        result = subprocess.run(['nvcc', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            # Parse nvcc output
            for line in result.stdout.split('\n'):
                if 'release' in line.lower():
                    parts = line.split('release')
                    if len(parts) > 1:
                        version = parts[1].strip().split(',')[0].strip()
                        cuda_info['nvcc_version'] = version
        else:
            cuda_info['nvcc_error'] = 'nvcc execution failed'
    except FileNotFoundError:
        cuda_info['nvcc_error'] = 'nvcc not found'
    except subprocess.TimeoutExpired:
        cuda_info['nvcc_error'] = 'nvcc execution timeout'
    except Exception as e:
        cuda_info['nvcc_error'] = f'Error getting CUDA info: {str(e)}'
    
    # Try to get from environment variables
    import os
    cuda_path = os.environ.get('CUDA_PATH', '')
    if cuda_path:
        cuda_info['cuda_path'] = cuda_path
    
    return cuda_info

def get_pytorch_info():
    """Get PyTorch information"""
    pytorch_info = {}
    try:
        import torch
        pytorch_info['available'] = True
        pytorch_info['version'] = torch.__version__
        pytorch_info['cuda_available'] = torch.cuda.is_available()
        
        if pytorch_info['cuda_available']:
            pytorch_info['cuda_version'] = torch.version.cuda
            pytorch_info['cudnn_version'] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
            pytorch_info['gpu_count'] = torch.cuda.device_count()
            pytorch_info['current_device'] = torch.cuda.current_device() if pytorch_info['gpu_count'] > 0 else None
            pytorch_info['device_name'] = torch.cuda.get_device_name(0) if pytorch_info['gpu_count'] > 0 else None
        else:
            pytorch_info['cuda_version'] = None
            pytorch_info['cudnn_version'] = None
            pytorch_info['gpu_count'] = 0
    except ImportError:
        pytorch_info['available'] = False
        pytorch_info['error'] = 'PyTorch not installed'
    except Exception as e:
        pytorch_info['available'] = False
        pytorch_info['error'] = f'Error getting PyTorch info: {str(e)}'
    
    return pytorch_info

def get_dataloader_info():
    """Get DataLoader multiprocessing information"""
    dataloader_info = {}
    try:
        import torch
        import torch.multiprocessing as mp
        
        dataloader_info['multiprocessing_available'] = True
        dataloader_info['multiprocessing_method'] = mp.get_start_method(allow_none=True)
        dataloader_info['cpu_count'] = mp.cpu_count() if hasattr(mp, 'cpu_count') else None
        
        # Check if multiprocessing can be used
        try:
            # On Windows, spawn is the default method
            if platform.system() == 'Windows':
                dataloader_info['multiprocessing_supported'] = True
                dataloader_info['recommended_workers'] = 0  # Usually recommended to use 0 on Windows
            else:
                dataloader_info['multiprocessing_supported'] = True
                dataloader_info['recommended_workers'] = min(4, mp.cpu_count() if mp.cpu_count() else 0)
        except Exception as e:
            dataloader_info['multiprocessing_supported'] = False
            dataloader_info['error'] = str(e)
            
    except ImportError:
        dataloader_info['multiprocessing_available'] = False
        dataloader_info['error'] = 'PyTorch not installed, cannot check multiprocessing support'
    except Exception as e:
        dataloader_info['error'] = f'Error getting DataLoader info: {str(e)}'
    
    return dataloader_info

def main():
    """Main function"""
    print("=" * 80)
    print("环境信息收集")
    print("=" * 80)
    print()
    
    # 1. Operating system information
    print("【1. 操作系统信息】")
    os_info = get_os_info()
    print(f"  操作系统: {os_info['platform']}")
    print(f"  系统名称: {os_info['os_name']}")
    print(f"  系统版本: {os_info['os_version']}")
    print(f"  系统架构: {os_info['os_arch']}")
    print()
    
    # 2. GPU information
    print("【2. GPU信息】")
    gpu_info = get_gpu_info()
    if 'gpus' in gpu_info:
        print(f"  GPU数量: {gpu_info['gpu_count']}")
        for i, gpu in enumerate(gpu_info['gpus']):
            print(f"  GPU {i}: {gpu['name']}")
            print(f"    驱动版本: {gpu['driver_version']}")
    else:
        print(f"  {gpu_info.get('error', '未知错误')}")
    print()
    
    # 3. CUDA information
    print("【3. CUDA信息】")
    cuda_info = get_cuda_info()
    if 'nvcc_version' in cuda_info:
        print(f"  CUDA版本 (nvcc): {cuda_info['nvcc_version']}")
    else:
        print(f"  nvcc: {cuda_info.get('nvcc_error', '未找到')}")
    if 'cuda_path' in cuda_info:
        print(f"  CUDA路径: {cuda_info['cuda_path']}")
    print()
    
    # 4. PyTorch information
    print("【4. PyTorch信息】")
    pytorch_info = get_pytorch_info()
    if pytorch_info.get('available', False):
        print(f"  PyTorch已安装: 是")
        print(f"  PyTorch版本: {pytorch_info['version']}")
        print(f"  CUDA可用: {'是' if pytorch_info['cuda_available'] else '否'}")
        if pytorch_info['cuda_available']:
            print(f"  PyTorch CUDA版本: {pytorch_info['cuda_version']}")
            if pytorch_info.get('cudnn_version'):
                print(f"  cuDNN版本: {pytorch_info['cudnn_version']}")
            print(f"  GPU数量 (PyTorch检测): {pytorch_info['gpu_count']}")
            if pytorch_info['gpu_count'] > 0:
                print(f"  当前设备: {pytorch_info['current_device']}")
                print(f"  设备名称: {pytorch_info['device_name']}")
    else:
        print(f"  PyTorch已安装: 否")
        print(f"  错误: {pytorch_info.get('error', '未知错误')}")
    print()
    
    # 5. DataLoader multiprocessing information
    print("【5. DataLoader多进程信息】")
    dataloader_info = get_dataloader_info()
    if dataloader_info.get('multiprocessing_available', False):
        print(f"  多进程支持: 是")
        print(f"  多进程方法: {dataloader_info.get('multiprocessing_method', '未知')}")
        if 'cpu_count' in dataloader_info and dataloader_info['cpu_count']:
            print(f"  CPU核心数: {dataloader_info['cpu_count']}")
        if 'recommended_workers' in dataloader_info:
            print(f"  推荐workers数: {dataloader_info['recommended_workers']}")
        if 'multiprocessing_supported' in dataloader_info:
            print(f"  多进程可用: {'是' if dataloader_info['multiprocessing_supported'] else '否'}")
    else:
        print(f"  多进程支持: 否")
        print(f"  错误: {dataloader_info.get('error', '未知错误')}")
    print()
    
    print("=" * 80)
    print("信息收集完成")
    print("=" * 80)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
