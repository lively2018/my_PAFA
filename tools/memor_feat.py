import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import time

from numpy import log
from tools.demo import make_parser
# Load the data
df = pd.read_csv('/home/kssong/memory_size_result/memory_stats_min.csv')

levels = ['P3', 'P4', 'P5']
norm_factors = {'P3': 6400, 'P4': 1600, 'P5': 400}
level_colors = {'P3': 'tab:blue', 'P4': 'tab:orange', 'P5': 'tab:green'}
MEMORY_LIMIT = 4800
RATIO_LIMIT = 1.0
P3_LIMIT = 6400
P4_LIMIT = 1600
P5_LIMIT = 400
def make_parser():
    parser = argparse.ArgumentParser("Memory feat video")
    parser.add_argument("--path", type=str, default="/home/kssong/memory_size_result/memory_bank_stats_video.csv",\
                         help="path to the CSV file containing memory bank stats")
    return parser

def main(args):
    current_time = time.localtime()
    start_time = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    log_file_path = os.path.join("./", f'log_{start_time}.txt')
    log_file = open(log_file_path, 'w')
    df = pd.read_csv(args.path)
    # 1. Feature Number Trace - Separated by Level
    fig, axes = plt.subplots(len(levels), 1, figsize=(10, 10), sharex=True)
    for i, level in enumerate(levels):
        subset = df[df['level'] == level]
        axes[i].plot(subset['frame_idx'], subset['count'], label=f'Level {level}', color=level_colors[level], linewidth=1.5)
        axes[i].set_title(f'Feature Number Trace - Level {level}')
        axes[i].set_ylim(0, norm_factors[level])
        axes[i].set_ylabel('Count')
        axes[i].legend(loc='upper right')
        axes[i].grid(True, linestyle='--', alpha=0.6)

    axes[-1].set_xlabel('Frame Index')
    plt.tight_layout()
    plt.savefig(f'feature_growth_sep_{start_time}.png')

    # 2. Feature Ratio Trace - Separated by Level
    fig, axes = plt.subplots(len(levels), 1, figsize=(10, 10), sharex=True)
    for i, level in enumerate(levels):
        subset = df[df['level'] == level]
        ratios = subset['count'] / norm_factors[level]
        axes[i].plot(subset['frame_idx'], ratios, label=f'Level {level}', color=level_colors[level], linewidth=1.5)
        axes[i].set_ylim(0, RATIO_LIMIT)
        axes[i].set_title(f'Feature Ratio Trace - Level {level}')
        axes[i].set_ylabel('Ratio')
        axes[i].legend(loc='upper right')
        axes[i].grid(True, linestyle='--', alpha=0.6)

    axes[-1].set_xlabel('Frame Index')
    plt.tight_layout()
    plt.savefig(f'feature_ratio_grow_sep_{start_time}.png')

    # 3. Memory Usage Trace - Separated by Level
    fig, axes = plt.subplots(len(levels), 1, figsize=(10, 10), sharex=True)
    for i, level in enumerate(levels):
        subset = df[df['level'] == level]
        axes[i].plot(subset['frame_idx'], subset['gpu_mem_mb'], label=f'Level {level}', color=level_colors[level], linewidth=1.5)
        axes[i].set_ylim(0, MEMORY_LIMIT)
        axes[i].set_title(f'Memory Usage Trace - Level {level}')
        axes[i].set_ylabel('Memory')
        axes[i].legend(loc='upper right')
        axes[i].grid(True, linestyle='--', alpha=0.6)

    axes[-1].set_xlabel('Frame Index')
    plt.tight_layout()
    plt.savefig(f'memory_usage_grow_sep_{start_time}.png')

    level_info = {}
    for level in levels:
        subset = df[df['level'] == level]
        total_count = subset['count'].sum()
        average_count = subset['count'].mean()
        level_info[level] = {'total_count': total_count, 'average_count': average_count}
        level_info[level]['average_ratio'] = average_count / norm_factors[level]
        level_info[level]['max_count'] = subset['count'].max()
        level_info[level]['max_ratio'] = subset['count'].max() / norm_factors[level]
        level_info[level]['min_count'] = subset['count'].min()
        level_info[level]['min_ratio'] = subset['count'].min() / norm_factors[level]
        level_info[level]['percentage_average'] = (average_count / norm_factors[level]) * 100
        level_info[level]['percentage_max'] = (subset['count'].max() / norm_factors[level]) * 100
        level_info[level]['percentage_min'] = (subset['count'].min() / norm_factors[level]) * 100

    for level in level_info.keys():
        print(f'Level {level} - Average Features Number: {level_info[level]["average_count"]:.2f}')
        print(f'Level {level} - Average Ratio: {level_info[level]["average_ratio"]:.3f}')
        print(f'Level {level} - Max Features Number: {level_info[level]["max_count"]:.2f}')
        print(f'Level {level} - Max_Ratio: {level_info[level]["max_ratio"]:.2f}')
        print(f'Level {level} - Min Features Number: {level_info[level]["min_count"]:.2f}')
        print(f'Level {level} - Min_Ratio: {level_info[level]["min_ratio"]:.2f}')
        print(f'Level {level} - Percentage of Average Feature Number: {level_info[level]["percentage_average"]:.2f}%')
        print(f'Level {level} - Percentage of Max Features Number: {level_info[level]["percentage_max"]:.2f}%')
        print(f'Level {level} - Percentage of Min Features Number: {level_info[level]["percentage_min"]:.2f}%')
        log_file.write(f'Level {level} - Average Features Number: {level_info[level]["average_count"]:.2f}\n')
        log_file.write(f'Level {level} - Average Ratio: {level_info[level]["average_ratio"]:.3f}\n')
        log_file.write(f'Level {level} - Max Features Number: {level_info[level]["max_count"]:.2f}\n')
        log_file.write(f'Level {level} - Max_Ratio: {level_info[level]["max_ratio"]:.2f}\n')
        log_file.write(f'Level {level} - Min Features Number: {level_info[level]["min_count"]:.2f}\n')
        log_file.write(f'Level {level} - Min_Ratio: {level_info[level]["min_ratio"]:.2f}\n')
        log_file.write(f'Level {level} - Percentage of Average Feature Number: {level_info[level]["percentage_average"]:.2f}%\n')
        log_file.write(f'Level {level} - Percentage of Max Features Number: {level_info[level]["percentage_max"]:.2f}%\n')
        log_file.write(f'Level {level} - Percentage of Min Features Number: {level_info[level]["percentage_min"]:.2f}%\n')

    level_mem_info = {}
    for level in levels:
        subset = df[df['level'] == level]
        level_mem_info[level] = {}
        level_mem_info[level]['total_memory_feature'] = subset['gpu_mem_mb'].sum()
        level_mem_info[level]['average_memory_feature'] = subset['gpu_mem_mb'].mean()
        level_mem_info[level]['max_memory_feature'] = subset['gpu_mem_mb'].max()
        level_mem_info[level]['min_memory_feature'] = subset['gpu_mem_mb'].min()
        level_mem_info[level]['ratio_average_memory'] = subset['gpu_mem_mb'].mean() / MEMORY_LIMIT
        level_mem_info[level]['ratio_max_memory'] = subset['gpu_mem_mb'].max() / MEMORY_LIMIT
        level_mem_info[level]['ratio_min_memory'] = subset['gpu_mem_mb'].min() / MEMORY_LIMIT
        level_mem_info[level]['percentage_average_memory'] = (subset['gpu_mem_mb'].mean() / MEMORY_LIMIT) * 100
        level_mem_info[level]['percentage_max_memory'] = (subset['gpu_mem_mb'].max() / MEMORY_LIMIT) * 100
        level_mem_info[level]['percentage_min_memory'] = (subset['gpu_mem_mb'].min() / MEMORY_LIMIT) * 100
    for level in level_mem_info.keys():
        print(f'Level {level} - Average Memory Feature: {level_mem_info[level]["average_memory_feature"]:.2f}')
        print(f'Level {level} - Max Memory Feature: {level_mem_info[level]["max_memory_feature"]:.2f}')
        print(f'Level {level} - Min Memory Feature: {level_mem_info[level]["min_memory_feature"]:.2f}')
        print(f'Level {level} - Ratio of Average Memory Feature: {level_mem_info[level]["ratio_average_memory"]:.3f}')
        print(f'Level {level} - Ratio of Max Memory Feature: {level_mem_info[level]["ratio_max_memory"]:.3f}')
        print(f'Level {level} - Ratio of Min Memory Feature: {level_mem_info[level]["ratio_min_memory"]:.3f}')
        print(f'Level {level} - Percentage of Average Memory Feature: {level_mem_info[level]["percentage_average_memory"]:.2f}%')
        print(f'Level {level} - Percentage of Max Memory Feature: {level_mem_info[level]["percentage_max_memory"]:.2f}%')
        print(f'Level {level} - Percentage of Min Memory Feature: {level_mem_info[level]["percentage_min_memory"]:.2f}%')
        log_file.write(f'Level {level} - Average Memory Feature: {level_mem_info[level]["average_memory_feature"]:.2f}\n')
        log_file.write(f'Level {level} - Max Memory Feature: {level_mem_info[level]["max_memory_feature"]:.2f}\n')
        log_file.write(f'Level {level} - Min Memory Feature: {level_mem_info[level]["min_memory_feature"]:.2f}\n')
        log_file.write(f'Level {level} - Ratio of Average Memory Feature:   {level_mem_info[level]["ratio_average_memory"]:.3f}\n')
        log_file.write(f'Level {level} - Ratio of Max Memory Feature: {level_mem_info[level]["ratio_max_memory"]:.3f}\n')
        log_file.write(f'Level {level} - Ratio of Min Memory Feature: {level_mem_info[level]["ratio_min_memory"]:.3f}\n')
        log_file.write(f'Level {level} - Percentage of Average Memory Feature: {level_mem_info[level]["percentage_average_memory"]:.2f}%\n')
        log_file.write(f'Level {level} - Percentage of Max Memory Feature: {level_mem_info[level]["percentage_max_memory"]:.2f}%\n')
        log_file.write(f'Level {level} - Percentage of Min Memory Feature: {level_mem_info[level]["percentage_min_memory"]:.2f}%\n')
    log_file.close()
if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)