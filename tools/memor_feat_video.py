import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('/home/kssong/memory_size_result/memory_bank_stats_video.csv')

levels = ['P3', 'P4', 'P5']
norm_factors = {'P3': 6400, 'P4': 1600, 'P5': 400}
level_colors = {'P3': 'tab:blue', 'P4': 'tab:orange', 'P5': 'tab:green'}
MEMORY_LIMIT = 4800
RATIO_LIMIT = 1.0
PERCENTAGE_LIMIT = 100.0
P3_LIMIT = 6400
P4_LIMIT = 1600
P5_LIMIT = 400

average_video_info = {}
#video_path', 'batch_set_count', 'level', 'count', 'mem_len'
# 1. Feature Number Trace - Separated by Level and video
video_count = 0
for video_name in df['video_path'].unique():
    video_count += 1
    video_df = df[df['video_path'] == video_name]

    average_video_info[video_count] = {'video_num': video_count, 'video_name': video_name}
    for i, level in enumerate(levels):
        subset = video_df[video_df['level'] == level]
        average_count = subset['count'].mean()
        average_ratio = average_count / (norm_factors[level] * 16)
        average_percentage = average_ratio * 100
        average_mem_len = subset['mem_len'].mean()
        average_mem_ratio = average_mem_len / MEMORY_LIMIT
        average_mem_percentage = average_mem_ratio * 100
        average_video_info[video_count].update({
            f'average_count_{level}': average_count,
            f'average_ratio_{level}': average_ratio,
            f'average_percentage_{level}': average_percentage,
            f'average_mem_len_{level}': average_mem_len,
            f'average_mem_ratio_{level}': average_mem_ratio,
            f'average_mem_percentage_{level}': average_mem_percentage
        })

total_video_count = len(average_video_info)
for level in levels:
    total_average_count = sum(info[f'average_count_{level}'] for info in average_video_info.values())
    total_average_mem_len = sum(info[f'average_mem_len_{level}'] for info in average_video_info.values())
    total_average_count_per_video = total_average_count / (total_video_count)
    total_average_mem_len_per_video = total_average_mem_len / (total_video_count)
    total_average_percentage_per_video = total_average_count_per_video / (norm_factors[level]*16) * 100
    total_average_mem_percentage_per_video = total_average_mem_len_per_video / (MEMORY_LIMIT) * 100
    print(f"  {level} Total Average Count: {total_average_count:.2f} Average Total Average Count: {total_average_count_per_video:.2f}")
    print(f"  {level} Total Average Memory Length: {total_average_mem_len:.2f} Average Total Average Memory Length: {total_average_mem_len_per_video:.2f}")
    print(f"  {level} Total Average Percentage: {(total_average_percentage_per_video):.3f}%")
    print(f"  {level} Total Average Memory Percentage: {(total_average_mem_percentage_per_video):.3f}%")
    print(f"  {level} Max Count: {max(info[f'average_count_{level}'] for info in average_video_info.values())} video(s) with max count: {[info['video_name'] for info in average_video_info.values() if info[f'average_count_{level}'] == max(info[f'average_count_{level}'] for info in average_video_info.values())]} ")

for _, info in average_video_info.items():
    print(f"Video {info['video_num']}: {info['video_name']}")
    for level in levels:
        print(f"  Average Count {level}: {info[f'average_count_{level}']:.2f}")
        print(f"  Average Ratio {level}: {info[f'average_ratio_{level}']:.4f} ({info[f'average_percentage_{level}']:.2f}%)")
        print(f"  Average Memory Length: {info[f'average_mem_len_{level}']:.2f}")
        print(f"  Average Memory Ratio: {info[f'average_mem_ratio_{level}']:.4f} ({info[f'average_mem_percentage_{level}']:.2f}%)")

# 2. Feature Count - Separated by Level (one subplot per level, one point per video)
fig, axes = plt.subplots(len(levels), 1, figsize=(10, 10), sharex=True)
for i, level in enumerate(levels):
    video_nums = [v['video_num'] for v in average_video_info.values()]
    counts = [v[f'average_count_{level}'] for v in average_video_info.values()]
    axes[i].plot(video_nums, counts, label=f'Level {level}', color=level_colors[level], linewidth=1.5, marker='o')
    axes[i].set_ylim(0, norm_factors[level] * 16)
    axes[i].set_title(f'Feature Numbers Trace - Level {level}')
    axes[i].set_ylabel('Feature Count')
    axes[i].legend(loc='upper right')
    axes[i].grid(True, linestyle='--', alpha=0.6)
    axes[i].set_xlabel('Video Number')
axes[-1].set_xlabel('Video Number')
plt.tight_layout()
plt.savefig('Feature_Numbers_Trace_by_Level_Video.png')
fig, axes = plt.subplots(len(levels), 1, figsize=(10, 10), sharex=True)
for i, level in enumerate(levels):
    video_nums = [v['video_num'] for v in average_video_info.values()]
    counts = [v[f'average_percentage_{level}'] for v in average_video_info.values()]
    print(f"Level {level} - Max Percentage: {max(counts):.2f}% video(s) with max percentage: {[v['video_name'] for v in average_video_info.values() if v[f'average_percentage_{level}'] == max(counts)]}")
    axes[i].plot(video_nums, counts, label=f'Level {level}', color=level_colors[level], linewidth=1.5, marker='o')
    axes[i].set_ylim(0, PERCENTAGE_LIMIT)
    axes[i].set_title(f'Feature Percentages Trace - Level {level}')
    axes[i].set_ylabel('Feature Percentage (%)')
    axes[i].legend(loc='upper right')
    axes[i].grid(True, linestyle='--', alpha=0.6)
    axes[i].set_xlabel('Video Number')

plt.tight_layout()
plt.savefig('Feature_Percentage_Trace_by_Level_Video.png')
# 3. Memory Usage - Separated by Level (one subplot per level, one point per video)
fig, axes = plt.subplots(len(levels), 1, figsize=(10, 10), sharex=True)
for i, level in enumerate(levels):
    video_nums = [v['video_num'] for v in average_video_info.values()]
    mem_lens = [v[f'average_mem_len_{level}'] for v in average_video_info.values()]
    axes[i].plot(video_nums, mem_lens, label=f'Level {level}', color=level_colors[level], linewidth=1.5, marker='o')
    axes[i].set_ylim(0, MEMORY_LIMIT)
    axes[i].set_title(f'Memory Usage Trace - Level {level}')
    axes[i].set_ylabel('Memory')
    axes[i].legend(loc='upper right')
    axes[i].grid(True, linestyle='--', alpha=0.6)
    axes[i].set_xlabel('Video Number')

plt.tight_layout()
plt.savefig('Memory_Usage_Trace_by_Level_Video.png')

fig, axes = plt.subplots(len(levels), 1, figsize=(10, 10), sharex=True)
for i, level in enumerate(levels):
    video_nums = [v['video_num'] for v in average_video_info.values()]
    mem_lens = [v[f'average_mem_percentage_{level}'] for v in average_video_info.values()]
    axes[i].plot(video_nums, mem_lens, label=f'Level {level}', color=level_colors[level], linewidth=1.5, marker='o')
    axes[i].set_ylim(0, PERCENTAGE_LIMIT)
    axes[i].set_title(f'Memory Usage Percentage Trace - Level {level}')
    axes[i].set_ylabel('Memory Percentage (%)')
    axes[i].legend(loc='upper right')
    axes[i].grid(True, linestyle='--', alpha=0.6)
    axes[i].set_xlabel('Video Number')

plt.tight_layout()
plt.savefig('Memory_Usage_Percentage_Trace_by_Level_Video.png')