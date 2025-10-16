from tqdm.auto import tqdm
import os
from collections import Counter
import rasterio
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from scipy.optimize import minimize


# ------------------------------------------------------------------------------
# 定义标准化方法
class StdScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def transform(self, array, mode, original_data=None):
        if mode == 0:  # 标准化
            self.mean = np.mean(array)
            self.std = np.std(array)

            if self.std == 0:
                print("警告: 标准差为0，数据无变化")
                return array - self.mean  # 标准差为零时返回0中心化的数组

            return (array - self.mean) / self.std

        elif mode == 1:  # 还原
            if self.mean is None or self.std is None:
                # 如果没有进行标准化，且未提供原始数据，则无法还原
                if original_data is None:
                    raise ValueError("未进行标准化，也未提供原始数据进行还原")

                # 使用原始数据计算中介值
                print("未进行标准化，正在使用原始数据计算中介值进行还原")
                self.mean = np.mean(original_data)
                self.std = np.std(original_data)

            return array * self.std + self.mean

        else:
            raise ValueError("无效的 mode 值，mode 只能是 0（标准化）或 1（还原）")


# 特征矩阵提取
def eigenmatrix(ci_abundances):
    # 获取所有可能的类别（通过查看所有字典中的键）
    categories = sorted(set(category for abundance in ci_abundances for category in abundance.keys()))

    # 初始化丰度矩阵
    abundance_matrix = np.zeros((len(ci_abundances), len(categories)))

    # 填充丰度矩阵
    for i, abundance in enumerate(ci_abundances):
        for category, value in abundance.items():
            col_index = categories.index(category)
            abundance_matrix[i, col_index] = value

    # 创建类别标签映射字典，位置对应，值初始化为空
    class_mapping = {category: None for category in categories}

    return abundance_matrix, class_mapping


# 定义目标函数
def loss_function(s_mi, ci_eigenmatrix, s_mi_p, erpha, klass_number, mi_num):
    # 计算正则化项的常数，避免在目标函数中重复计算
    reg_weight = erpha * mi_num / klass_number

    # 定义目标函数
    def objective(bar_S):
        # 计算预测值，使用权重矩阵和待求特征值
        predictions = np.dot(ci_eigenmatrix, bar_S)

        # 计算残差平方和，衡量预测值与实际信号值之间的差异
        rss = np.sum((s_mi - predictions) ** 2)

        # 计算正则化项，衡量待求特征值与预设特征值之间的差异
        regularization = np.sum((bar_S - s_mi_p) ** 2)  # 使用预设的特征值 S'

        # 返回总损失值，包括残差平方和和正则化项
        return rss + reg_weight * regularization

    # 优化任务封装函数
    def run_optimization(initial_guess):
        try:
            # 使用 BFGS 方法进行优化，无约束条件
            result = minimize(objective, initial_guess, method='BFGS', options={'disp': False})
            return result.x, result.fun  # 返回优化结果和目标函数值
        except Exception as e:
            print(f"优化失败: {e}")
            return None, None

    # 生成随机初始值
    initial_guesses = s_mi_p + np.random.randn(klass_number) * 0.05
    x_opt, fun_value = run_optimization(initial_guesses)
    # print(f'$$$$$$$$$$误差大小：{fun_value}')

    # 输出最终的优化结果
    return x_opt


# 对目标函数再封装
def unmix_calculation(mi_pixels, ci_abundances,
                      klass_number,
                      mi_prime, mi_num,
                      erpha):
    s_mi = np.array(mi_pixels, dtype=np.float32)  # 实际观测值

    abundance_matrix, class_mapping = eigenmatrix(ci_abundances)

    ci_eigenmatrix = abundance_matrix

    s_mi_p = np.array(mi_prime, dtype=np.float32)

    s_mi_hat = loss_function(s_mi, ci_eigenmatrix, s_mi_p, erpha, klass_number, mi_num)

    for i, class_label in enumerate(class_mapping.keys()):
        class_mapping[class_label] = s_mi_hat[i]  # 为 class_mapping 赋值

    # print(f"————————丰度矩阵————————")
    # print(ci_eigenmatrix)
    # print(f"————————解混初结果————————")
    # print(s_mi_hat)
    # print(f"————————解混类字典————————")
    # print(class_mapping)
    # print(f"————————误差字典————————")

    # 返回类别映射和误差项
    return class_mapping


# 从给定的小窗口中提取 MI 像素值和 CI 丰度数据（以比例形式存储）
def extract_mi_and_abundance_win(window_mi,
                                 window_ci,
                                 window_size,
                                 fake_wndo_size):
    fake_wndo_step = fake_wndo_size  # 滑动窗口步长

    # 计算小窗口的数量
    num_windows_y = (window_size - fake_wndo_size) // fake_wndo_step + 1
    num_windows_x = (window_size - fake_wndo_size) // fake_wndo_step + 1

    # 创建小窗口索引数组
    y_indices = np.arange(0, num_windows_y * fake_wndo_step, fake_wndo_step)
    x_indices = np.arange(0, num_windows_x * fake_wndo_step, fake_wndo_step)

    # 生成小窗口切片
    mi_windows = np.array([window_mi[i:i + fake_wndo_size, j:j + fake_wndo_size]
                           for i in y_indices for j in x_indices])
    ci_windows = np.array([window_ci[i:i + fake_wndo_size, j:j + fake_wndo_size]
                           for i in y_indices for j in x_indices])

    # 提取 MI 矩阵的代表值（每个小窗口的第一个像素值）
    mi_pixels = mi_windows[:, 0, 0]  # 使用第一个像素代表整个窗口

    # 计算 CI 丰度比例
    ci_abundances = []
    for ci_window in ci_windows:
        # 展平小窗口并统计每个类别的数量
        flattened_window = ci_window.flatten()
        fake_category_counts = Counter(flattened_window)

        # 计算总元素数量
        total_elements = np.sum(list(fake_category_counts.values()))

        # 计算每个类别的比例
        if total_elements > 0:
            fake_category_ratios = {k: v / total_elements for k, v in fake_category_counts.items()}
        else:
            fake_category_ratios = {}  # 如果总数为 0，返回空字典

        # 存储结果
        ci_abundances.append(fake_category_ratios)

    return mi_pixels, ci_abundances


# 提取每个类别的 MI 图像区域并计算该区域的中位数
def extract_category_medians_win(window_mi,
                                 window_ci,
                                 bigcategory_counts):
    # 获取所有类别的唯一值和对应的掩码
    unique_categories = np.array(list(bigcategory_counts.keys()))
    masks = [(window_ci == category) for category in unique_categories]

    # 提取 MI 图像中对应每个类别的像素值
    category_pixels_list = [window_mi[mask] for mask in masks]

    # 计算每个类别的中位数
    mi_prime = np.array([
        np.median(category_pixels) if category_pixels.size > 0 else np.nan
        for category_pixels in category_pixels_list
    ])

    # 检查是否存在没有像素值的类别
    for category, pixels in zip(unique_categories, category_pixels_list):
        if pixels.size == 0:
            print(f"类别 '{category}' 在当前窗口没有对应的像素值。")

    return mi_prime


# 记录输出定位坐标与窗口结果矩阵
def location_and_replace_win(i, j,
                             window_size,
                             fake_wndo_size,
                             original_ci,
                             every_window_result,
                             original_high, original_width):
    # 计算大窗口的中心坐标
    center_window_x = i + window_size // 2
    center_window_y = j + window_size // 2
    fake_wndo_half = fake_wndo_size // 2
    start_x = center_window_x - fake_wndo_half
    start_y = center_window_y - fake_wndo_half

    # 检查窗口是否超出矩阵边界
    if (start_x < 0 or start_y < 0
            or (start_x + fake_wndo_size) > original_high or (start_y + fake_wndo_size) > original_width):
        return None, None  # 如果窗口超出边界，返回 None

    # 获取窗口中的标签数据
    fake_wndo_labels = original_ci[start_x:start_x + fake_wndo_size,
                       start_y:start_y + fake_wndo_size]

    # 创建空的窗口矩阵
    local_result = np.zeros((fake_wndo_size, fake_wndo_size), dtype=np.float32)

    # 将结果字典转换为数组（如果标签是整数且范围有限，推荐使用此方法）
    max_label = max(every_window_result.keys())
    result_array = np.full(max_label + 1, np.nan, dtype=np.float32)  # 初始化默认值为 NaN
    for label, value in every_window_result.items():
        result_array[label] = value

    # 矢量化替换操作
    valid_mask = fake_wndo_labels <= max_label  # 检查标签是否有效（避免越界）
    local_result[valid_mask] = result_array[fake_wndo_labels[valid_mask]]

    # 检查未替换的标签
    invalid_mask = ~valid_mask  # 无效标签的位置
    if np.any(invalid_mask):
        invalid_labels = np.unique(fake_wndo_labels[invalid_mask])
        print(f"以下标签未找到在结果中，无法替换：{invalid_labels}")

    return start_x, start_y, local_result


# 预处理数据：优化I/O操作，提前加载图像和ci数据
def load_data(image_mi, image_ci):
    # 读取遥感影像
    with rasterio.open(image_mi) as dataset:
        original_mi = dataset.read(1)
        # 标准化处理
        scaler = StdScaler()
        original_mi_Std = scaler.transform(original_mi, mode=0)
    with rasterio.open(image_ci) as dataset:
        original_ci = dataset.read(1)

    return original_mi_Std, original_ci


# 定义函数图像恢复
def image_restore(image_path, finallys, output_path):
    try:
        # 检查文件是否存在
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"原始图像文件未找到: {image_path}")

        # 读取原始影像数据
        with rasterio.open(image_path) as src:
            image_data = src.read(1)  # 读取第一个波段
            meta = src.meta.copy()  # 获取影像的元数据

        # 确保 CSV 数据与影像数据的形状匹配
        if finallys.shape != image_data.shape:
            raise ValueError("CSV 数据的形状与影像数据不匹配。")

        # 更新元数据，指定数据类型为 float64，并使用 GeoTIFF 格式
        meta.update({
            'dtype': 'float64',
            'count': 1,
            'driver': 'GTiff'  # 确保使用 GTiff 格式
        })
        scaler = StdScaler()
        result_final_unstd = scaler.transform(finallys, mode=1, original_data=image_data)
        # 写入新的影像数据
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(result_final_unstd.astype('float64'), 1)

        print(f"影像已成功保存到: {output_path}")

    except Exception as e:
        print(f"错误: {e}")


# 统计大窗口中每个类别的像素数量，以及类别数量
def count_win(window_ci):
    # 统计大窗口中每个类别的像素数量
    bigcategory_counts = Counter(window_ci.astype(np.uint8).flatten())

    # 统计类别数量
    klass_number = len(bigcategory_counts)

    return bigcategory_counts, klass_number


def process_window_batch(batch_indices, original_mi_Std, original_ci, window_size, fake_wndo_size, mi_num, elpha):
    local_results = []

    # 批量提取窗口
    windows_mi = np.array([original_mi_Std[i:i + window_size, j:j + window_size] for i, j in batch_indices])
    windows_ci = np.array([original_ci[i:i + window_size, j:j + window_size] for i, j in batch_indices])

    # 遍历每个窗口
    for window_mi, window_ci, (i, j) in zip(windows_mi, windows_ci, batch_indices):
        bigcategory_counts, klass_number = count_win(window_ci)
        mi_pixels, ci_abundances = extract_mi_and_abundance_win(window_mi, window_ci, window_size, fake_wndo_size)
        mi_prime = extract_category_medians_win(window_mi, window_ci, bigcategory_counts)
        # -----------------------------------------------
        every_window_result = unmix_calculation(mi_pixels, ci_abundances, klass_number, mi_prime, mi_num, elpha)
        start_x, start_y, local_result = location_and_replace_win(i, j, window_size, fake_wndo_size, original_ci,
                                                                  every_window_result, original_mi_Std.shape[0],
                                                                  original_mi_Std.shape[1])
        local_results.append((start_x, start_y, local_result))

    return local_results


def slide_window(image_mi, image_ci, window_size, fake_wndo_size, elpha):
    try:
        print(f"开始处理滑动窗口操作，图像路径：{image_mi}，CSV 路径：{image_ci}")
        window_step = fake_wndo_size
        original_mi_Std, original_ci = load_data(image_mi, image_ci)
        original_high, original_width = original_mi_Std.shape
        print(f"图像和 CSV 数据尺寸: {original_high}x{original_width}")

        result_final = np.zeros_like(original_mi_Std)  # 初始化结果矩阵
        mi_num = int((window_size / window_step) * (window_size / window_step))

        # 设置进度条
        total_tasks = ((original_high - window_size) // window_step + 1) * (
                (original_width - window_size) // window_step + 1)
        progress_bar = tqdm(total=total_tasks, desc="Processing", unit="window")

        # 获取系统的CPU核心数（使用最大核心数来实现满功率）
        # max_workers = os.cpu_count()
        max_workers = 3

        # 批量生成所有窗口的起始坐标
        x_indices = np.arange(0, original_high - window_size + 1, window_step)
        y_indices = np.arange(0, original_width - window_size + 1, window_step)
        grid_x, grid_y = np.meshgrid(x_indices, y_indices, indexing="ij")
        batch_indices = np.column_stack((grid_x.ravel(), grid_y.ravel()))

        # 将窗口按批次划分
        batch_size = total_tasks // (max_workers * 6)
        # batch_size = total_tasks // (max_workers)
        batches = [batch_indices[i:i + batch_size] for i in range(0, len(batch_indices), batch_size)]

        # 并行处理批次
        futures = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for batch in batches:
                futures.append(
                    executor.submit(process_window_batch, batch, original_mi_Std, original_ci, window_size,
                                    fake_wndo_size,
                                    mi_num, elpha)
                )

            # 等待并更新进度条
            for future in as_completed(futures):
                local_results = future.result()  # 获取批次结果

                # 提取批次的起始坐标和局部结果
                batch_indices = [(start_x, start_y) for start_x, start_y, _ in local_results]
                batch_local_results = np.array([result for _, _, result in local_results])

                # 批量更新全局结果矩阵
                matrix_summary_batch(result_final, batch_indices, batch_local_results, fake_wndo_size)

                # 更新进度条
                progress_bar.update(len(local_results))

        progress_bar.close()  # 关闭进度条
        return result_final

    except Exception as e:
        print(f"滑动窗口操作时发生错误: {e}")


# 优化后的 matrix_summary_batch 函数
def matrix_summary_batch(result_final, batch_indices, local_results, fake_wndo_size):
    for (start_x, start_y), local_result in zip(batch_indices, local_results):
        result_final[start_x:start_x + fake_wndo_size, start_y:start_y + fake_wndo_size] = local_result
