import os
import threading
import numpy as np
import ujson as json
import pandas as pd
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
from joblib import Parallel, delayed
import time
import psutil
import matplotlib
import traceback

# stop render GUI to interrupt main process
matplotlib.use('Agg')
os.environ['MPLBACKEND'] = 'Agg'

output_dir = r'C:\Users\ASUS\PycharmProjects\PythonProject\output'

# global control of monitor
cpu_usage = []
memory_usage = []
timestamps = []
monitoring = True

# interval control version
# def monitor_resources(interval=1):
#     while monitoring:
#         timestamp = time.strftime("%H:%M:%S", time.localtime())
#         cpu = psutil.cpu_percent(interval=interval)
#         memory = psutil.virtual_memory().percent
#         cpu_usage.append(cpu)
#         memory_usage.append(memory)
#         timestamps.append(timestamp)

# time sleep version
def monitor_resources(interval=0.5):
    while monitoring:
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        cpu_usage.append(psutil.cpu_percent(interval))
        memory_usage.append(psutil.virtual_memory().percent)
        timestamps.append(timestamp)
        time.sleep(1) # record per second

# monitor visualization
def plot_resource_usage(output_file):
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(timestamps, cpu_usage, label='CPU Usage (%)', color='tab:red')
    plt.xlabel('Time')
    plt.ylabel('CPU Usage (%)')
    plt.title('CPU Usage Over Time')
    plt.xticks(rotation=45)

    plt.subplot(2, 1, 2)
    plt.plot(timestamps, memory_usage, label='Memory Usage (%)', color='tab:blue')
    plt.xlabel('Time')
    plt.ylabel('Memory Usage (%)')
    plt.title('Memory Usage Over Time')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


# version generator using yield to load data
# def load_data(file_path):
#     try:
#         with open(file_path, 'r', encoding='utf-8') as f:
#             for line in f:
#                 try:
#                     doc = json.loads(line.strip())
#                     yield {
#                         'keywords': doc.get('keywords', []),
#                         'scores': doc.get('scores', []),
#                         'cluster': doc.get('cluster', [])
#                     }
#                 except json.JSONDecodeError as e:
#                     print(f"Error decoding JSON in {file_path}: {e}")
#     except Exception as e:
#         print(f"Error reading {file_path}: {e}")

# load data into format list-dict
def load_data(file_path):
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    doc = json.loads(line.strip())
                    data.append({
                        'keywords': doc.get('keywords', []),
                        'scores': doc.get('scores', []),
                        'cluster': doc.get('cluster', [])
                    })
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in {file_path}: {e}")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

    return data

def filter_keywords(keywords):
    return [keyword for keyword in keywords if re.match(r'^[a-zA-Z]+$', keyword)]

def get_top_keywords(data, top_n=100):
    all_keywords = []
    for entry in data:
        filtered_keywords = filter_keywords(entry['keywords'])
        all_keywords.extend(filtered_keywords) # gather all keywords

    if all_keywords:
        keyword_counts = Counter(all_keywords)
        top_keywords = keyword_counts.most_common(top_n)
    else:
        print("No keywords found.")
        top_keywords = []

    return top_keywords

def group_by_cluster(data):
    clustered_data = defaultdict(list)
    for entry in data:
        cluster = entry['cluster']
        clustered_data[cluster].append(entry)

    # find top 3 volume of clusters
    sorted_clusters = sorted(clustered_data.items(), key=lambda x: len(x[1]), reverse=True)[:3]

    return dict(sorted_clusters)


def save_top_keywords_to_csv(top_keywords, category_file, cluster):
    df = pd.DataFrame(top_keywords, columns=['Keyword', 'Average Score'])
    category_name = os.path.basename(category_file)
    output_csv = os.path.join(output_dir, f"{category_name.replace('.json', '')}_cluster_{cluster}_top_keywords.csv")
    df.to_csv(output_csv, index=False)
    print(f"Top keywords for cluster {cluster} saved to {output_csv}")

def generate_wordcloud(keywords, output_file):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(keywords))
    wordcloud.to_file(output_file) # save directly to avoid calling the render process

def plot_keyword_histogram(top_keywords, output_file):
    keywords, frequencies = zip(*top_keywords)
    colors = plt.cm.Blues(np.linspace(0.3, 1, len(frequencies)))  # 使用蓝色渐变
    plt.figure(figsize=(14, 8))
    plt.barh(keywords[:20], frequencies[:20], color=colors)
    plt.title(f'Top Keywords Frequency', fontsize=20, fontweight='bold', color='darkblue')
    plt.xlabel('Frequency', fontsize=16, color='black')
    plt.ylabel('Keyword', fontsize=16, color='black')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.gca().invert_yaxis()
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()

def process_cluster(cluster, docs, category_file):

    # print(f"Processing cluster {cluster} in file {category_file}")
    top_keywords = get_top_keywords(docs, top_n=100)

    save_top_keywords_to_csv(top_keywords, category_file, cluster)

    category_name = os.path.basename(category_file)
    wordcloud_file = os.path.join(output_dir, f"{category_name.replace('.json', '')}_cluster_{cluster}_wordcloud.png")
    histogram_file = os.path.join(output_dir, f"{category_name.replace('.json', '')}_cluster_{cluster}_histogram.png")

    # Step 5: visualization
    generate_wordcloud(top_keywords, wordcloud_file)
    # print(f"Word cloud for cluster {cluster} saved to {wordcloud_file}")
    plot_keyword_histogram(top_keywords, histogram_file)
    # print(f"Histogram for cluster {cluster} saved to {histogram_file}")

def process_file(category_file):
    # print(f"Processing file: {category_file}")

    start_time = time.time()
    # Step 2: load data in one Json file
    category_data = load_data(category_file)

    if category_data:
        # Step 3: grouped data by cluster
        clustered_data = group_by_cluster(category_data)

        # Step 4: parallel processing every cluster
        Parallel(n_jobs=-1)(
            delayed(process_cluster)(cluster, docs, category_file)
            for cluster, docs in clustered_data.items()
        )

        # Step 6: analysis of cs category
        top_keywords = get_top_keywords(category_data, top_n=100)

        histogram_file = os.path.join(output_dir, "category_histogram.png")
        wordcloud_file = os.path.join(output_dir, "category_wordcloud.png")

        if not top_keywords:
            print('empty:', category_file)
        else:
            plot_keyword_histogram(top_keywords, histogram_file)
            generate_wordcloud(top_keywords, wordcloud_file)

        save_top_keywords_to_csv(top_keywords, category_file, 'category')

    else:
        print(f"No data found in {category_file}")

    end_time = time.time()
    print(f"Time taken to process {category_file}: {end_time - start_time:.2f} seconds")


def main():
    def get_all_files_in_directory(datapath):
        file_paths = []
        for root, dirs, files in os.walk(datapath):
            for file in files:
                if file.endswith('.json'):
                    file_paths.append(os.path.join(root, file))
        return file_paths

    datapath = r'C:\Users\ASUS\PycharmProjects\PythonProject\ALL'
    file_paths = get_all_files_in_directory(datapath)

    # Step 0: create monitor thread
    monitor_thread = threading.Thread(target=monitor_resources, args=(0.1,), daemon=True)
    monitor_thread.start()

    # Step 1: start main process to process files in parallel
    try:
        start_time = time.time()

        Parallel(n_jobs=-1)(delayed(process_file)(file) for file in file_paths)

        # ensure recording the precise time
        end_time = time.time()

        #  stop monitoring
        global monitoring
        monitoring = False

        print(f"Total time taken: {end_time - start_time:.2f} seconds")

        # visualize for monitor
        monitor_thread.join()

        resource_file = os.path.join(output_dir, "resource_usage.png")
        plot_resource_usage(resource_file)
        print(timestamps,cpu_usage,memory_usage)
        # print(f"Resource usage visualization saved to {resource_file}")

    except Exception as e:
        print(f"Error during processing: {e}")
        traceback.print_exc()
        monitoring = False
        monitor_thread.join()


if __name__ == '__main__':
    main()
