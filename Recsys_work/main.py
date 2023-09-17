import os
import psutil
import csv
import datetime
import time

# 记录CSV文件的路径
csv_file_path = '/Users/jiaqi_zheng/Desktop/Coding/python/Recsys_git/Recsys_work/resource/execution_logs.csv'
def recommendation():
    # 获取当前时间
    current_datetime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # 记录程序开始时的内存使用
    initial_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # 转换为MB
    start_time = time.time()
    # 协同过滤算法
    import collaborativeFilteringAlgorithmRecommendation
    collaborativeFilteringAlgorithmRecommendation
    current_function = "Collaborative_Filtering_Algorithm_Recommendation"
    # 神经协同过滤系统
    # import neuralCollaborativeFiltering
    # neuralCollaborativeFiltering
    # current_function = "Neural_Collaborative_Filtering"
    # 基于内容的推荐算法
    # import contentBasedRecommendationAlgorithms
    # contentBasedRecommendationAlgorithms
    # current_function = "Content_Based_Recommendation_Algorithms"
    # 图神经网络推荐系统
    # import graphtest
    # current_function = "Graph_Neural_Network_Recommendation"
    end_time = time.time()
    # 计算代码执行时间
    execution_time = end_time - start_time
    print(f"代码执行时间：{execution_time:.6f} 秒")
    # 记录程序结束时的内存使用
    final_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # 转换为MB
    # 计算峰值内存使用
    peak_memory = final_memory - initial_memory
    print(f"程序峰值内存使用: {peak_memory:.2f} MB")
    # 将记录写入CSV文件
    with open(csv_file_path, 'a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([current_datetime, current_function, execution_time, peak_memory])

if __name__ == '__main__':
    recommendation()
