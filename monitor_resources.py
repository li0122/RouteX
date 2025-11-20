#!/usr/bin/env python3
"""
系統資源監控腳本
用於監控H100服務器在訓練過程中的資源使用情況
"""

import time
import subprocess
import psutil
import threading
from datetime import datetime

def get_gpu_info():
    """獲取GPU使用情況"""
    try:
        result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            gpu_info = []
            for i, line in enumerate(lines):
                parts = line.split(', ')
                if len(parts) >= 4:
                    gpu_util, mem_used, mem_total, temp = parts[:4]
                    gpu_info.append({
                        'gpu_id': i,
                        'utilization': int(gpu_util),
                        'memory_used': int(mem_used),
                        'memory_total': int(mem_total),
                        'temperature': int(temp)
                    })
            return gpu_info
    except Exception as e:
        print(f"獲取GPU資訊失敗: {e}")
    
    return []

def get_system_info():
    """獲取系統資源使用情況"""
    # CPU使用率
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_count = psutil.cpu_count()
    
    # 記憶體使用率
    memory = psutil.virtual_memory()
    memory_percent = memory.percent
    memory_used_gb = memory.used / (1024**3)
    memory_total_gb = memory.total / (1024**3)
    
    # 磁碟I/O
    disk_io = psutil.disk_io_counters()
    
    # 網路I/O
    net_io = psutil.net_io_counters()
    
    # 進程數
    process_count = len(psutil.pids())
    
    return {
        'cpu_percent': cpu_percent,
        'cpu_count': cpu_count,
        'memory_percent': memory_percent,
        'memory_used_gb': memory_used_gb,
        'memory_total_gb': memory_total_gb,
        'disk_read_gb': disk_io.read_bytes / (1024**3) if disk_io else 0,
        'disk_write_gb': disk_io.write_bytes / (1024**3) if disk_io else 0,
        'net_sent_gb': net_io.bytes_sent / (1024**3) if net_io else 0,
        'net_recv_gb': net_io.bytes_recv / (1024**3) if net_io else 0,
        'process_count': process_count
    }

def get_python_processes():
    """獲取Python相關進程"""
    python_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'cmdline']):
        try:
            if 'python' in proc.info['name'].lower():
                cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                if 'train_model.py' in cmdline or 'RouteX' in cmdline:
                    python_processes.append({
                        'pid': proc.info['pid'],
                        'name': proc.info['name'],
                        'cpu_percent': proc.info['cpu_percent'],
                        'memory_percent': proc.info['memory_percent'],
                        'cmdline': cmdline[:100] + '...' if len(cmdline) > 100 else cmdline
                    })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    return python_processes

def monitor_resources(duration_minutes=30, interval_seconds=10):
    """監控系統資源"""
    print(f"開始監控系統資源 (持續 {duration_minutes} 分鐘，每 {interval_seconds} 秒更新)")
    print("=" * 80)
    
    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)
    
    # 記錄最大值
    max_values = {
        'cpu_percent': 0,
        'memory_percent': 0,
        'gpu_utilization': 0,
        'gpu_memory_percent': 0
    }
    
    try:
        while time.time() < end_time:
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            # 獲取系統資訊
            sys_info = get_system_info()
            gpu_info = get_gpu_info()
            python_procs = get_python_processes()
            
            # 更新最大值
            max_values['cpu_percent'] = max(max_values['cpu_percent'], sys_info['cpu_percent'])
            max_values['memory_percent'] = max(max_values['memory_percent'], sys_info['memory_percent'])
            
            # 清屏並顯示資訊
            print("\\033[2J\\033[H")  # 清屏
            print(f"時間: {timestamp} | 監控中... (Ctrl+C 停止)")
            print("=" * 80)
            
            # 系統資源
            print(f"CPU: {sys_info['cpu_percent']:5.1f}% ({sys_info['cpu_count']} cores) | "
                  f"記憶體: {sys_info['memory_percent']:5.1f}% "
                  f"({sys_info['memory_used_gb']:.1f}/{sys_info['memory_total_gb']:.1f} GB)")
            
            # GPU資源
            if gpu_info:
                for gpu in gpu_info:
                    gpu_mem_percent = (gpu['memory_used'] / gpu['memory_total']) * 100
                    max_values['gpu_utilization'] = max(max_values['gpu_utilization'], gpu['utilization'])
                    max_values['gpu_memory_percent'] = max(max_values['gpu_memory_percent'], gpu_mem_percent)
                    
                    print(f"GPU{gpu['gpu_id']}: {gpu['utilization']:3d}% | "
                          f"記憶體: {gpu_mem_percent:5.1f}% "
                          f"({gpu['memory_used']}/{gpu['memory_total']} MB) | "
                          f"溫度: {gpu['temperature']}°C")
            else:
                print("GPU: 無法獲取資訊")
            
            # Python進程
            print("\\nPython 訓練進程:")
            if python_procs:
                for proc in python_procs:
                    print(f"  PID {proc['pid']}: CPU {proc['cpu_percent']:5.1f}% | "
                          f"記憶體 {proc['memory_percent']:5.1f}% | {proc['cmdline']}")
            else:
                print("  無相關Python進程")
            
            # 最大值統計
            print("\\n最大使用率:")
            print(f"  CPU: {max_values['cpu_percent']:.1f}% | "
                  f"記憶體: {max_values['memory_percent']:.1f}% | "
                  f"GPU: {max_values['gpu_utilization']:.1f}% | "
                  f"GPU記憶體: {max_values['gpu_memory_percent']:.1f}%")
            
            # 資源利用率評估
            print("\\n資源利用率評估:")
            if max_values['cpu_percent'] < 50:
                print("  ️  CPU利用率較低，可考慮增加並行worker數量")
            if max_values['memory_percent'] < 50:
                print("  ️  記憶體利用率較低，可考慮增加批次大小或資料量")
            if max_values['gpu_utilization'] < 70:
                print("  ️  GPU利用率較低，可考慮增大模型或批次大小")
            
            print("=" * 80)
            
            time.sleep(interval_seconds)
            
    except KeyboardInterrupt:
        print("\\n監控已停止")
    
    # 最終報告
    print("\\n最終資源使用報告:")
    print(f"最高CPU使用率: {max_values['cpu_percent']:.1f}%")
    print(f"最高記憶體使用率: {max_values['memory_percent']:.1f}%")
    print(f"最高GPU使用率: {max_values['gpu_utilization']:.1f}%")
    print(f"最高GPU記憶體使用率: {max_values['gpu_memory_percent']:.1f}%")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='系統資源監控')
    parser.add_argument('--duration', type=int, default=30, help='監控持續時間(分鐘)')
    parser.add_argument('--interval', type=int, default=10, help='更新間隔(秒)')
    
    args = parser.parse_args()
    
    monitor_resources(args.duration, args.interval)

if __name__ == "__main__":
    main()