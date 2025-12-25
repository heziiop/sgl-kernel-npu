import re
import sys

def analyze_npu_memory(file_path):
    # 正则匹配 ptr, size, result
    pattern = re.compile(r"ptr=(0x[0-9a-fA-F]+),\s+size=(\d+),\s+result=(\d+)")

    allocations = []

    try:
        with open(file_path, 'r') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    ptr_str, size_str, result = match.groups()
                    if result == "0": # 只统计成功的分配
                        start = int(ptr_str, 16)
                        size = int(size_str)
                        allocations.append({
                            'start': start,
                            'end': start + size,
                            'size': size,
                            'ptr': ptr_str
                        })
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到")
        return

    if not allocations:
        print("未发现有效分配记录")
        return

    # 按地址排序
    allocations.sort(key=lambda x: x['start'])

    total_allocated_size = sum(a['size'] for a in allocations)
    min_addr = allocations[0]['start']
    max_addr = allocations[-1]['end']
    total_span = max_addr - min_addr  # 内存跨度（从第一个块开头到最后一个块结尾）

    total_gap_size = 0
    overlap_count = 0

    print(f"--- 内存分析报告 ---")
    print(f"记录总数: {len(allocations)}")
    print(f"总分配大小: {total_allocated_size / 1024**2:.2f} MB")
    print(f"内存地址跨度: {total_span / 1024**2:.2f} MB")

    for i in range(len(allocations) - 1):
        curr = allocations[i]
        next_a = allocations[i+1]

        if curr['end'] > next_a['start']:
            # 发现重叠
            overlap_count += 1
            print(f"❌ OVERLAP: {curr['ptr']} 与 {next_a['ptr']} 重叠 {curr['end'] - next_a['start']} bytes")
        else:
            # 统计块与块之间的间隙（即碎片空间）
            gap = next_a['start'] - curr['end']
            total_gap_size += gap

    # 碎片占比计算：间隙总和 / 总跨度
    # 注意：这里的碎片指外部碎片（External Fragmentation）
    frag_ratio = (total_gap_size / total_span) * 100 if total_span > 0 else 0

    print(f"-" * 30)
    print(f"检测到重叠处: {overlap_count}")
    print(f"块间总空隙 (碎片): {total_gap_size / 1024**2:.2f} MB")
    print(f"外部碎片占比: {frag_ratio:.2f}%")
    print(f"提示: 碎片占比高意味着内存不连续，可能导致大对象分配失败。")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze_npu_memory(sys.argv[1])
    else:
        print("用法: python script.py <logfile>")
        analyze_npu_memory("run.log")
