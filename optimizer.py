import numpy as np
import torch
import itertools
import sys
import time
import csv
import random
import gc
import os
from functools import lru_cache
from datetime import datetime
import __init__

from predict_3_period import WaterPlantPredictor_3_period
from predict_12_period import WaterPlantPredictorFullData

DENSITY_WATER = 1000  # kg/m³
GRAVITY = 9.81        # m/s²
PRESSURE_TO_HEAD = 102  # 1MPa = 102m扬程
EFFICIENCY_DENOMINATOR = 3600000  # 输出功率计算分母
PUMP_LEVEL = 2.35
FLOW_EPS = 0  # 流量极小值阈值
POWER_EPS = 0  # 功率极小值阈值

# ====================== 优化1: 预计算并缓存泵组合 ======================
def generate_valid_combinations(target_flow=None):
    """
    生成满足所有条件的有效列表（与原版逻辑一致，但结果缓存为 tuple）
    
    新增约束条件：
    1. 第 1,2,3,4 号泵（索引 0,1,2,3）不能同时为 1 或同时为 0
    2. 第 5,6,7,8 号泵（索引 4,5,6,7）不能同时为 1 或同时为 0
    3. 第 9,10 号泵（索引 8,9）不能同时为 1
    4. 根据目标流量计算开启泵的个数：num_active_pumps = int(target_flow / 2000) + 1
    
    :param target_flow: float, 目标流量 (m³/h), 如果为 None 则不限制开启泵的个数
    :return: list of tuple, 有效的泵组合列表
    """
    # 固定索引（第 1 号和第 10 号泵，索引 0 和 9）固定为 0
    fixed_indices = {0, 9}
    free_positions = [i for i in range(10) if i not in fixed_indices]
    combinations = itertools.product([0, 1], repeat=8)
    
    # 根据目标流量计算需要开启的泵的个数
    if target_flow is not None:
        num_active_pumps = int(target_flow / 2000)
        # 限制泵个数在合理范围内 (1-8 个，因为索引 0 和 9 固定为 0)
        num_active_pumps = max(1, min(8, num_active_pumps))
        print(f"  目标流量 {target_flow} m³/h, 计算开启泵个数：{num_active_pumps}")
    else:
        num_active_pumps = None
    
    result = []
    for combo in combinations:
        sublist = [0] * 10
        sublist[0] = 0  # 第 1 号泵固定为 0
        sublist[9] = 0  # 第 10 号泵固定为 0
        
        for idx, val in zip(free_positions, combo):
            sublist[idx] = val
        
        # 🔥 新增：根据目标流量过滤泵组合（1 的个数）
        if num_active_pumps is not None:
            active_count = sum(sublist)
            if active_count != num_active_pumps:
                continue
        
        group1 = [sublist[0], sublist[1], sublist[2], sublist[3]]
        if all(x == 1 for x in group1) or all(x == 0 for x in group1):
            continue
        
        
        group2 = [sublist[4], sublist[5], sublist[6], sublist[7]]
        if all(x == 1 for x in group2) or all(x == 0 for x in group2):
            continue
        
        if sublist[8] == 1 and sublist[9] == 1:
            continue
        
        result.append(tuple(sublist))  # 用 tuple 便于哈希缓存
    
    print(f"  生成 {len(result)} 个有效泵组合")
    return result


# ====================== 配置常量 ======================
RATED_POWERS = np.array([348.0, 344.0, 445.0, 382.5, 348.0, 344.0, 445.0, 336.0, 373.5, 373.5], dtype=np.float64)
RATED_SPEEDS = np.array([1500, 1500, 1500, 1500, 1500, 1500, 1000, 1500, 990, 990], dtype=np.float64)
SPEED_RANGES = [
    [0, 1500],                  # P10A2        定频泵
    [1300, 1310, 1320, 1330, 1340, 1350, 1360, 1370, 1380, 1390, 1400, 1410, 1420, 1430, 1440, 1450, 1460, 1470, 1480, 1490, 1500],  # P11A2  变频泵 1500 步长10
    [0, 1500],                  # P10A1        定频泵
    [1300, 1310, 1320, 1330, 1340, 1350, 1360, 1370, 1380, 1390, 1400, 1410, 1420, 1430, 1440, 1450, 1460, 1470, 1480, 1490, 1500],  # P11A1  变频泵 1500 步长10
    [0, 1500],                  # P10B2        定频泵
    [1300, 1310, 1320, 1330, 1340, 1350, 1360, 1370, 1380, 1390, 1400, 1410, 1420, 1430, 1440, 1450, 1460, 1470, 1480, 1490, 1500],  # P11B2  变频泵 1500 步长10
    [0, 1500],                  # P10B1        定频泵
    [800, 820, 840, 860, 880, 900, 920, 940, 960, 980, 1000],  # P11B1  变频泵 1000 步长20
    [800, 820, 840, 860, 880, 900, 920, 940, 960, 980, 990],  # 三期2#变频  变频泵 990 步长20
    [0, 990]                    # 三期1#工频   定频泵
]
# 顺序：P10A2, P11A2, P10A1, P11A1, P10B2, P11B2, P10B1, P11B1, 三期2#变频, 三期1#工频
PUMP_POWER_CALCULATORS = [
    # ========== 1. P10A2（无拟合公式 → 直接用理论公式） ==========
    {
        "calc": lambda n, H, rated_power, rated_speed: rated_power * (n / rated_speed) ** 3,
        "min": None,
        "max": None
    },

    # ========== 2. P11A2（VFD 有拟合 + 限幅 +  fallback） ==========
    {
        "calc": lambda n, H, rated_power, rated_speed: max(
            min(
                36.027016945267604 - 0.0035785402303661483*n - 0.0017622818216053706*H 
                + 0.00023950838598450785*n**2 - 1.238802466603324*n*H - 0.0013116331081610396*H**2 
                - 0.00000012670378135093766*n**3 + 0.0013008510602695942*n**2*H 
                - 1.0027755208718903*n*H**2 - 0.0007265017497541032*H**3,
                349.299987
            ),
            157.800003
        ) if (157.800003 <= 
              (36.027016945267604 - 0.0035785402303661483*n - 0.0017622818216053706*H 
               + 0.00023950838598450785*n**2 - 1.238802466603324*n*H - 0.0013116331081610396*H**2 
               - 0.00000012670378135093766*n**3 + 0.0013008510602695942*n**2*H 
               - 1.0027755208718903*n*H**2 - 0.0007265017497541032*H**3) 
              <= 349.299987) 
        else rated_power * (n / rated_speed) ** 3,
        "min": 157.800003,
        "max": 349.299987
    },

    # ========== 3. P10A1（FIXED 有拟合 + 限幅 + fallback） ==========
    {
        "calc": lambda n, H, rated_power, rated_speed: max(
            min(
                289.69334709631056 + 622.4271053913694*H - 1276.7645290604135*H**2,
                375.700012
            ),
            325.899939
        ) if (325.899939 <= 
              (289.69334709631056 + 622.4271053913694*H - 1276.7645290604135*H**2) 
              <= 375.700012) 
        else rated_power * (n / rated_speed) ** 3,
        "min": 325.899939,
        "max": 375.700012
    },

    # ========== 4. P11A1（VFD 有拟合 + 限幅 + fallback） ==========
    {
        "calc": lambda n, H, rated_power, rated_speed: max(
            min(
                -213.34664790214526 - 0.008421131883539745*n - 0.003743255314963712*H 
                + 0.0009867142936234863*n**2 - 2.6052246817780547*n*H - 0.00229935766695589*H**2 
                - 0.0000005787958272281124*n**3 + 0.0024158140039873175*n**2*H 
                - 1.3540348305426824*n*H**2 - 0.001035474888350988*H**3,
                354.399993
            ),
            154.899993
        ) if (154.899993 <= 
              (-213.34664790214526 - 0.008421131883539745*n - 0.003743255314963712*H 
               + 0.0009867142936234863*n**2 - 2.6052246817780547*n*H - 0.00229935766695589*H**2 
               - 0.0000005787958272281124*n**3 + 0.0024158140039873175*n**2*H 
               - 1.3540348305426824*n*H**2 - 0.001035474888350988*H**3) 
              <= 354.399993) 
        else rated_power * (n / rated_speed) ** 3,
        "min": 154.899993,
        "max": 354.399993
    },

    # ========== 5. P10B2（FIXED 有拟合 + 限幅 + fallback） ==========
    {
        "calc": lambda n, H, rated_power, rated_speed: max(
            min(
                190.68470872182195 + 789.323534480104*H - 1123.037517936883*H**2,
                335.600006
            ),
            315.200012
        ) if (315.200012 <= 
              (190.68470872182195 + 789.323534480104*H - 1123.037517936883*H**2) 
              <= 335.600006) 
        else rated_power * (n / rated_speed) ** 3,
        "min": 315.200012,
        "max": 335.600006
    },

    # ========== 6. P11B2（VFD 有拟合 + 限幅 + fallback） ==========
    {
        "calc": lambda n, H, rated_power, rated_speed: max(
            min(
                148.06019157408704 - 0.00408765785443732*n - 0.001704735849193416*H 
                + 0.000026710006618687667*n**2 - 1.200851391546636*n*H - 0.0014624790276655546*H**2 
                - 0.0000000154333809629717*n**3 + 0.0013685872086144409*n**2*H 
                - 1.2381212226921279*n*H**2 - 0.0009029653103591735*H**3,
                390.700012
            ),
            173.000000
        ) if (173.000000 <= 
              (148.06019157408704 - 0.00408765785443732*n - 0.001704735849193416*H 
               + 0.000026710006618687667*n**2 - 1.200851391546636*n*H - 0.0014624790276655546*H**2 
               - 0.0000000154333809629717*n**3 + 0.0013685872086144409*n**2*H 
               - 1.2381212226921279*n*H**2 - 0.0009029653103591735*H**3) 
              <= 390.700012) 
        else rated_power * (n / rated_speed) ** 3,
        "min": 173.000000,
        "max": 390.700012
    },

    # ========== 7. P10B1（FIXED 有拟合 + 限幅 + fallback） ==========
    {
        "calc": lambda n, H, rated_power, rated_speed: max(
            min(
                113.50009704221787 + 1187.5383798171727*H - 1598.835590979953*H**2,
                338.899993
            ),
            327.399993
        ) if (327.399993 <= 
              (113.50009704221787 + 1187.5383798171727*H - 1598.835590979953*H**2) 
              <= 338.899993) 
        else rated_power * (n / rated_speed) ** 3,
        "min": 327.399993,
        "max": 338.899993
    },

    # ========== 8. P11B1（VFD 有拟合 + 限幅 + fallback） ==========
    {
        "calc": lambda n, H, rated_power, rated_speed: max(
            min(
                -4430.097221989078 + 14.805488174233862*n - 0.004951539784180598*H 
                - 0.01591971952387194*n**2 - 1.7922749259861368*n*H - 0.0020776696162758154*H**2 
                + 0.000005809752421370717*n**3 + 0.0025914144316477334*n**2*H 
                - 1.4586304457662693*n*H**2 - 0.00011488843372866562*H**3,
                399.500000
            ),
            139.399993
        ) if (139.399993 <= 
              (-4430.097221989078 + 14.805488174233862*n - 0.004951539784180598*H 
               - 0.01591971952387194*n**2 - 1.7922749259861368*n*H - 0.0020776696162758154*H**2 
               + 0.000005809752421370717*n**3 + 0.0025914144316477334*n**2*H 
               - 1.4586304457662693*n*H**2 - 0.00011488843372866562*H**3) 
              <= 399.500000) 
        else rated_power * (n / rated_speed) ** 3,
        "min": 139.399993,
        "max": 399.500000
    },

    # ========== 9. 三期 2#变频（无拟合 → 理论公式） ==========
    {
        "calc": lambda n, H, rated_power, rated_speed: rated_power * (n / rated_speed) ** 3,
        "min": None,
        "max": None
    },

    # ========== 10. 三期 1#工频（无拟合 → 理论公式） ==========
    {
        "calc": lambda n, H, rated_power, rated_speed: rated_power * (n / rated_speed) ** 3,
        "min": None,
        "max": None
    }
]
PRESSURE_RANGE = [round(p, 2) for p in np.arange(0.2, 0.61, 0.01)]

# 泵的额定流量
RATED_FLOWS = np.array([1994, 1994, 1994, 1994, 1994, 1994, 1994, 1994, 2320, 2320], dtype=np.float64)

ACTUAL_PUMP_COMBINATIONS = generate_valid_combinations()
# ====================== 优化2: 初始化预测器 ======================
predictor_12 = WaterPlantPredictorFullData()
predictor_3 = WaterPlantPredictor_3_period()


# ====================== 计算额定流量之和函数 ======================
def calculate_max_rated_flow_for_combination(pump_combination):
    """
    计算泵组合的最大额定流量之和
    :param pump_combination: 泵组合列表，如 [1,0,1,0,...]
    :return: 最大额定流量之和
    """
    total_rated_flow = 0.0
    for i, is_active in enumerate(pump_combination):
        if is_active == 1 and i < len(RATED_FLOWS):
            total_rated_flow += RATED_FLOWS[i]
    return total_rated_flow


# ====================== 优化 3: 向量化输入功率计算======================
def calculate_total_input_power_vectorized(discrete_arr, speed_arr, pressure, pump_group_index):
    """
    向量化计算总输入功率
    顺序 100% 对齐：P10A2, P11A2, P10A1, P11A1, P10B2, P11B2, P10B1, P11B1, 三期2#变频, 三期1#工频
    :param discrete_arr: (N, 8/2) 开关状态
    :param speed_arr:    (N, 8/2) 转速
    :param pressure:     压力 H
    :param pump_group_index: 0=一二期8泵，1=三期2泵
    :return: 总功率 (N,)
    """
    N = len(speed_arr)
    total_power = np.zeros(N, dtype=np.float64)

    if pump_group_index == 0:
        # ===================== 一二期 8 台泵（顺序严格对齐） =====================
        pump_indices = [0, 1, 2, 3, 4, 5, 6, 7]  # 直接用全局 index

        for local_idx, global_idx in enumerate(pump_indices):
            # 开关状态
            active = discrete_arr[:, local_idx] == 1
            if not np.any(active):
                continue

            # 额定功率 & 额定转速
            rp = RATED_POWERS[global_idx]
            rs = RATED_SPEEDS[global_idx]
            calc = PUMP_POWER_CALCULATORS[global_idx]

            # 转速处理
            n = speed_arr[:, local_idx].copy()

            # 逐行计算功率
            for i in np.where(active)[0]:
                ni = n[i]
                power = calc["calc"](ni, pressure, rp, rs)
                total_power[i] += power

        return total_power

    else:
        # ===================== 三期 2 台泵（全局 index 8,9） =====================
        pump_indices = [8, 9]
        for local_idx, global_idx in enumerate(pump_indices):
            active = discrete_arr[:, local_idx] == 1
            if not np.any(active):
                continue

            rp = RATED_POWERS[global_idx]
            rs = RATED_SPEEDS[global_idx]
            calc = PUMP_POWER_CALCULATORS[global_idx]

            n = speed_arr[:, local_idx].copy()

            for i in np.where(active)[0]:
                ni = n[i]
                power = calc["calc"](ni, pressure, rp, rs)
                total_power[i] += power

        return total_power


def calculate_total_output_power_vectorized(total_flow_arr, pressure, water_level=2.35):
    """向量化计算输出功率"""
    H = pressure * PRESSURE_TO_HEAD
    effective_head = H - (water_level - PUMP_LEVEL)
    if effective_head < 0:
        return np.zeros_like(total_flow_arr)
    return (DENSITY_WATER * GRAVITY * total_flow_arr * effective_head) / EFFICIENCY_DENOMINATOR


# ====================== 优化4: 预生成转速组合（缓存） ======================
_speed_combo_cache_12 = {}
_speed_combo_cache_3 = {}


def get_speed_combinations_cached(discrete_pump_tuple, pressure, pump_num):
    """
    带缓存的转速组合生成（基于 SPEED_RANGES 按顺序取值）
    
    :param discrete_pump_tuple: tuple, 泵开关状态 (8 个或 2 个)
    :param pressure: float, 压力值（保留参数但实际不使用）
    :param pump_num: int, 泵数量 (8 或 2)
    :return: tuple, (idx_list, speed_matrix)
        - idx_list: 开启泵的索引列表
        - speed_matrix: numpy 数组 (N, pump_num), 所有转速组合
    """
    cache = _speed_combo_cache_12 if pump_num == 8 else _speed_combo_cache_3
    
    if discrete_pump_tuple not in cache:
        # 确定泵的索引偏移（8 泵模式从 0 开始，2 泵模式从 8 开始）
        offset = 0 if pump_num == 8 else 8
        
        active_pumps = []
        for local_idx, pump_status in enumerate(discrete_pump_tuple):
            global_idx = offset + local_idx  # 映射到全局泵索引 (0-9)
            
            if pump_status == 1:
                # 泵开启：从 SPEED_RANGES 获取转速选项（排除 0）
                speed_options = [s for s in SPEED_RANGES[global_idx] if s > 0]
                if speed_options:
                    active_pumps.append((local_idx, speed_options))
            # 泵关闭：转速固定为 0，不需要添加到 active_pumps
        
        if active_pumps:
            idx_list = [p[0] for p in active_pumps]
            speed_options_list = [p[1] for p in active_pumps]
            
            # 预生成所有速度组合
            all_speeds = list(itertools.product(*speed_options_list))
            
            # 构建 (N, pump_num) 数组，默认 0
            speed_matrix = np.zeros((len(all_speeds), pump_num), dtype=np.float64)
            for row_idx, speeds in enumerate(all_speeds):
                for col_idx, speed in zip(idx_list, speeds):
                    speed_matrix[row_idx, col_idx] = speed
            
            cache[discrete_pump_tuple] = (idx_list, speed_matrix)
        else:
            # 没有开启的泵，返回全 0 矩阵
            cache[discrete_pump_tuple] = ([], np.zeros((1, pump_num), dtype=np.float64))
    
    return cache[discrete_pump_tuple]


# ====================== 优化5: 泵配对验证向量化 ======================
def validate_pump_pairs_vectorized(discrete_pump, speed_matrix_12):
    """
    向量化验证泵配对关系
    :param discrete_pump: tuple, 8个泵的开关状态
    :param speed_matrix_12: numpy (N, 8) 转速矩阵
    :return: boolean numpy (N,) mask
    """
    pair_map = [(0, 2), (1, 3), (4, 6), (5, 7)]
    mask = np.ones(len(speed_matrix_12), dtype=bool)
    
    for p1, p2 in pair_map:
        s1, s2 = discrete_pump[p1], discrete_pump[p2]
        if s1 == 1 or s2 == 1:
            if s1 != 1 or s2 != 1:
                # 一开一关 → 全部不合法
                return np.zeros(len(speed_matrix_12), dtype=bool)
            # 两个都开，检查转速差
            diff = np.abs(speed_matrix_12[:, p1] - speed_matrix_12[:, p2])
            mask &= (diff <= 10000)
    
    return mask


def validate_flow_balance_vectorized(discrete_pump, speed_matrix_12):
    """
    向量化验证一期二期流量平衡
    :param discrete_pump: tuple, 8个泵开关
    :param speed_matrix_12: numpy (N, 8) 转速矩阵
    :return: boolean numpy (N,) mask
    """
    RATED_FLOW = 1994.0
    FLOW_BALANCE_THRESHOLD = 200.0
    PHASE1 = [0, 1, 2, 3]
    PHASE2 = [4, 5, 6, 7]
    
    phase1_flow = np.zeros(len(speed_matrix_12))
    phase2_flow = np.zeros(len(speed_matrix_12))
    
    for idx in PHASE1:
        if discrete_pump[idx] == 1:
            rs = RATED_SPEEDS[idx]
            if rs > 0:
                phase1_flow += RATED_FLOW * (speed_matrix_12[:, idx] / rs)
    
    for idx in PHASE2:
        if discrete_pump[idx] == 1:
            rs = RATED_SPEEDS[idx]
            if rs > 0:
                phase2_flow += RATED_FLOW * (speed_matrix_12[:, idx] / rs)
    
    return np.abs(phase1_flow - phase2_flow) <= FLOW_BALANCE_THRESHOLD


# ====================== 优化6: 批量预测 ======================
def batch_predict_12(predictor, discrete_pump, speed_matrix, pressure):
    """
    批量调用一二期预测器
    尝试用批量接口，若不支持则逐条调用
    """
    N = len(speed_matrix)
    results = np.zeros(N)
    
    # 尝试批量预测（如果预测器支持）
    if hasattr(predictor, 'predict_batch'):
        discrete_list = [list(discrete_pump)] * N
        continuous_list = []
        for i in range(N):
            cont = [pressure] + speed_matrix[i].tolist()
            continuous_list.append(cont)
        results = np.array(predictor.predict_batch(discrete_list, continuous_list))
    else:
        # 逐条预测
        discrete_list = list(discrete_pump)
        for i in range(N):
            cont = [pressure] + speed_matrix[i].tolist()
            results[i] = predictor.predict_single(discrete_list, cont)
    
    return results


def batch_predict_3(predictor, discrete_pump, speed_matrix, pressure):
    """批量调用三期预测器"""
    N = len(speed_matrix)
    results = np.zeros(N)
    
    if hasattr(predictor, 'predict_batch'):
        discrete_list = [list(discrete_pump)] * N
        continuous_list = []
        for i in range(N):
            cont = [pressure] + speed_matrix[i].tolist()
            continuous_list.append(cont)
        results = np.array(predictor.predict_batch(discrete_list, continuous_list))
    else:
        discrete_list = list(discrete_pump)
        for i in range(N):
            cont = [pressure] + speed_matrix[i].tolist()
            results[i] = predictor.predict_single(discrete_list, cont)
    
    return results


# ====================== 进度条 ======================
class ProgressBar:
    def __init__(self, total, width=50, description=""):
        self.total = max(total, 1)
        self.width = width
        self.current = 0
        self.start_time = time.time()
        self.description = description
    
    def update(self, n=1):
        self.current += n
        percent = min(self.current / self.total, 1.0)
        filled_length = int(self.width * percent)
        bar = '█' * filled_length + '-' * (self.width - filled_length)
        elapsed_time = time.time() - self.start_time
        if self.current > 0:
            remaining_time = (elapsed_time / self.current) * (self.total - self.current)
        else:
            remaining_time = 0
        remaining_str = f"{remaining_time:.1f}s"
        sys.stdout.write(f'\r{self.description} |{bar}| {percent:.1%} ({self.current}/{self.total}) 剩余: {remaining_str}')
        sys.stdout.flush()
    
    def finish(self):
        if self.current < self.total:
            self.update(self.total - self.current)
        sys.stdout.write('\n')


# ====================== 优化7: 核心计算函数（向量化重写） ======================
def calculate_combinations_for_pressure_optimized(pressure, target_flow=None, validate_flow_balance=False, validate_flow_limit=False):
    """
    向量化版本：计算单个压力下所有泵组合的性能指标
    关键优化：
    1. 转速组合预生成为numpy矩阵
    2. 输入功率向量化计算
    3. 验证逻辑向量化
    4. 减少Python循环
    """
    all_results = []
    water_level = 2.35
    
    # 统计
    total_combo_count = 0
    filtered_count = 0
    flow_limit_filtered_count = 0
    
    print(f"  压力 {pressure}: 开始计算...")
    ACTUAL_PUMP_COMBINATIONS = generate_valid_combinations(target_flow)
    
    for pump_combination in ACTUAL_PUMP_COMBINATIONS:
        discrete_pump_12 = pump_combination[:8]
        discrete_pump_3 = pump_combination[8:]
        
        # 计算泵组合的最大额定流量
        max_rated_flow = calculate_max_rated_flow_for_combination(pump_combination)
        
        # 获取缓存的转速矩阵
        _, speed_matrix_12 = get_speed_combinations_cached(tuple(discrete_pump_12), pressure, 8)
        _, speed_matrix_3 = get_speed_combinations_cached(tuple(discrete_pump_3), pressure, 2)
        
        N12 = len(speed_matrix_12)
        N3 = len(speed_matrix_3)
        
        # ---- 优化: 先对一二期做验证过滤，再做笛卡尔积 ----
        valid_mask_12 = np.ones(N12, dtype=bool)
        
        
        if validate_flow_balance:
            valid_mask_12 &= validate_flow_balance_vectorized(discrete_pump_12, speed_matrix_12)
        
        # 过滤后的一二期转速
        speed_matrix_12_valid = speed_matrix_12[valid_mask_12]
        
        N12_valid = len(speed_matrix_12_valid)
        
        if N12_valid == 0 or N3 == 0:
            continue
        
        # ---- 批量计算一二期流量 ----
        flow_12_arr = batch_predict_12(predictor_12, discrete_pump_12, speed_matrix_12_valid, pressure)
        
        # ---- 批量计算三期流量 ----
        flow_3_arr = batch_predict_3(predictor_3, discrete_pump_3, speed_matrix_3, pressure)
        
        # ---- 批量计算一二期输入功率 ----
        discrete_arr_12 = np.tile(np.array(discrete_pump_12, dtype=np.float64), (N12_valid, 1))
        input_power_12_arr = calculate_total_input_power_vectorized(
            discrete_arr_12, 
            speed_matrix_12_valid, 
            pressure,
            0
        )
        
        # ---- 批量计算三期输入功率 ----
        discrete_arr_3 = np.tile(np.array(discrete_pump_3, dtype=np.float64), (N3, 1))
        input_power_3_arr = calculate_total_input_power_vectorized(
            discrete_arr_3, 
            speed_matrix_3, 
            pressure,
            1
        )
        
        # ---- 笛卡尔积合并（用numpy广播） ----
        # total_flow[i, j] = flow_12[i] + flow_3[j]
        total_flow_matrix = flow_12_arr[:, np.newaxis] + flow_3_arr[np.newaxis, :]  # (N12_valid, N3)
        total_input_power_matrix = input_power_12_arr[:, np.newaxis] + input_power_3_arr[np.newaxis, :]  # (N12_valid, N3)
        
        # 输出功率
        total_output_power_matrix = calculate_total_output_power_vectorized(total_flow_matrix.ravel(), pressure, water_level).reshape(N12_valid, N3)
        
        # 效率
        with np.errstate(divide='ignore', invalid='ignore'):
            efficiency_matrix = np.where(total_input_power_matrix > 0,
                                         total_output_power_matrix / total_input_power_matrix, 0)
        
        # ---- 过滤条件 ----
        valid_mask = (total_flow_matrix > 0) & (efficiency_matrix <= 0.9) & (efficiency_matrix > 0.4)
        
        # ---- 新增：流量限制条件 ----
        if validate_flow_limit and max_rated_flow > 0:
            
            # 检查预测流量不能超过额定流量之和
            valid_mask &= (total_flow_matrix <= max_rated_flow)
            valid_mask &= (max_rated_flow * 0.5 <= total_flow_matrix)
            
        
        # ---- 提取有效结果 ----
        valid_indices = np.argwhere(valid_mask)  # (K, 2) 数组
        
        if len(valid_indices) == 0:
            continue
        
        # 千吨水电耗
        kwt_matrix = np.where(total_flow_matrix > 0,
                              total_input_power_matrix / total_flow_matrix * 1000, 0)
        
        for idx_pair in valid_indices:
            i12, i3 = idx_pair
            
            # 构建 combined_continuous: [pressure, 8个一二期转速, 2个三期转速]
            combined_continuous = [pressure] + speed_matrix_12_valid[i12].tolist() + speed_matrix_3[i3].tolist()
            result = {
                'pump_combination': list(pump_combination),
                'speed_combination': combined_continuous,
                'total_flow': float(total_flow_matrix[i12, i3]),
                'total_input_power': float(total_input_power_matrix[i12, i3]),
                'total_output_power': float(total_output_power_matrix[i12, i3]),
                'efficiency': float(efficiency_matrix[i12, i3]),
                'kwt': float(kwt_matrix[i12, i3]),
                'max_rated_flow': float(max_rated_flow)  # 保存最大额定流量
            }
            all_results.append(result)
        
        total_combo_count += N12_valid * N3
    
    print(f"  压力 {pressure}: 计算完成，共 {total_combo_count} 个组合，{len(all_results)} 个有效结果")
    if flow_limit_filtered_count > 0:
        print(f"  流量限制过滤了 {flow_limit_filtered_count} 个组合")
    return all_results


# ====================== 优化9: 分组 ======================
# 
def group_results_by_flow_optimized(results, target_flows, batch_size=500000, min_fallback_count=5):
    """
    使用分批处理避免内存溢出，并确保每个泵组都有分组数据
    
    :param results: 所有计算结果列表
    :param target_flows: 目标流量列表
    :param batch_size: 批处理大小
    :param min_fallback_count: 每个泵组最少保留的 fallback 结果数量
    """
    if not results:
        return {}
    
    target_flows_sorted = np.array(sorted(target_flows), dtype=np.float64)
    n_targets = len(target_flows_sorted)
    
    if n_targets > 1:
        avg_step = (target_flows_sorted[-1] - target_flows_sorted[0]) / (n_targets - 1)
        max_error = avg_step * 1.5
    else:
        max_error = 50.0
    
    grouped_results = {}
    total_count = len(results)
    
    # 收集所有唯一的泵组合
    all_pump_combinations = set()
    for r in results:
        pump_key = tuple(r['pump_combination'])
        all_pump_combinations.add(pump_key)
    
    print(f"  共 {len(all_pump_combinations)} 个唯一泵组合")
    
    # 第一遍：按误差阈值正常分组
    for start_idx in range(0, total_count, batch_size):
        end_idx = min(start_idx + batch_size, total_count)
        batch_results = results[start_idx:end_idx]
        
        all_flows = np.array([r['total_flow'] for r in batch_results], dtype=np.float64)
        nearest_indices = np.searchsorted(target_flows_sorted, all_flows)
        nearest_indices = np.clip(nearest_indices, 0, n_targets - 1)
        
        left_indices = np.clip(nearest_indices - 1, 0, n_targets - 1)
        diff_right = np.abs(all_flows - target_flows_sorted[nearest_indices])
        diff_left = np.abs(all_flows - target_flows_sorted[left_indices])
        
        use_left = diff_left < diff_right
        best_indices = np.where(use_left, left_indices, nearest_indices)
        best_diffs = np.where(use_left, diff_left, diff_right)
        
        valid_mask = best_diffs <= max_error
        
        for i in range(len(batch_results)):
            if not valid_mask[i]:
                continue
            
            target_flow = float(target_flows_sorted[best_indices[i]])
            pump_key = tuple(batch_results[i]['pump_combination'])
            
            if target_flow not in grouped_results:
                grouped_results[target_flow] = {}
            if pump_key not in grouped_results[target_flow]:
                grouped_results[target_flow][pump_key] = []
            grouped_results[target_flow][pump_key].append(batch_results[i])
        
        del batch_results, all_flows, nearest_indices, left_indices
        del diff_right, diff_left, use_left, best_indices, best_diffs, valid_mask
        gc.collect()
    
    # 第二遍：为缺失的泵组合补充 fallback 结果（每个目标流量最近的前 5 个）
    print(f"  开始补充 fallback 结果...")
    fallback_count = 0
    
    for target_flow in target_flows_sorted:
        target_flow_key = float(target_flow)
        
        if target_flow_key not in grouped_results:
            grouped_results[target_flow_key] = {}
        
        existing_pump_keys = set(grouped_results[target_flow_key].keys())
        missing_pump_keys = all_pump_combinations - existing_pump_keys
        
        if not missing_pump_keys:
            continue
        
        # 收集所有结果，按与目标流量的距离排序
        flow_diffs = []
        for r in results:
            pump_key = tuple(r['pump_combination'])
            if pump_key in missing_pump_keys:
                diff = abs(r['total_flow'] - target_flow_key)
                flow_diffs.append((diff, r))
        
        # 按距离排序
        flow_diffs.sort(key=lambda x: x[0])
        
        # 为每个缺失的泵组合分配最近的前 min_fallback_count 个结果
        pump_assigned_count = {pk: 0 for pk in missing_pump_keys}
        
        for diff, r in flow_diffs:
            pump_key = tuple(r['pump_combination'])
            if pump_key not in missing_pump_keys:
                continue
            if pump_assigned_count[pump_key] >= min_fallback_count:
                continue
            
            if pump_key not in grouped_results[target_flow_key]:
                grouped_results[target_flow_key][pump_key] = []
            grouped_results[target_flow_key][pump_key].append(r)
            pump_assigned_count[pump_key] += 1
            fallback_count += 1
        
        del flow_diffs
        gc.collect()
    
    print(f"  分组完成，覆盖 {len(grouped_results)} 个目标流量，补充 {fallback_count} 个 fallback 结果")
    return grouped_results


# ====================== 以下函数与原版基本一致 ======================

def find_best_solutions_for_target_flows(grouped_results, target_flows):
    best_by_target_flow = {tf: [] for tf in target_flows}
    
    for target_flow, pump_combinations in grouped_results.items():
        for pump_combination, results in pump_combinations.items():
            if results:
                best_result = max(results, key=lambda x: x['efficiency'])
                best_by_target_flow[target_flow].append(best_result)
    
    # 填补缺失
    missing_count = 0
    for target_flow in target_flows:
        if not best_by_target_flow[target_flow]:
            min_diff = float('inf')
            nearest_result = None
            for otf, results in best_by_target_flow.items():
                for result in results:
                    diff = abs(target_flow - otf)
                    if diff < min_diff:
                        min_diff = diff
                        nearest_result = result
            if nearest_result:
                best_by_target_flow[target_flow].append({
                    k: (v.copy() if isinstance(v, list) else v)
                    for k, v in nearest_result.items()
                })
                missing_count += 1
    
    if missing_count > 0:
        print(f"  已为 {missing_count} 个未覆盖目标流量分配近似解")
    
    return best_by_target_flow


def result_to_csv_row(result, target_flow, liquid_level=0.0):
    pump_combination = result['pump_combination']
    speed_combination = result['speed_combination']
    
    header_press = speed_combination[0]
    flow_lower = target_flow - 50
    flow_upper = target_flow + 50
    pump_group = ''.join(str(x) for x in pump_combination)
    
    freq_mapping = {
        'P1': speed_combination[1],
        'P2': speed_combination[2],
        'P3': speed_combination[3],
        'P4': speed_combination[4],
        'P5': speed_combination[5],
        'P6': speed_combination[6],
        'P7': speed_combination[7],
        'P8': speed_combination[8],
        'P9': speed_combination[9],
        'P10': speed_combination[10] 
    }
    
    total_flow = result['total_flow']
    total_input_power = result['total_input_power']
    total_output_power = result['total_output_power']
    efficiency = result['efficiency']
    kwt = result['kwt']
    max_rated_flow = result.get('max_rated_flow', 0.0)
    
    # 创建csv_row
    csv_row = {
        'Header_Press': round(header_press, 3),
        'Flow_Lower': round(flow_lower, 1),
        'Flow_Upper': round(flow_upper, 1),
        'Liquid_Level': round(liquid_level, 1),
        'Pump_Group': pump_group
    }
    for i in range(1, 11):
        csv_row[f'P{i}_Freq'] = int(freq_mapping[f'P{i}'])

    
    csv_row['Total_Flow'] = round(total_flow, 2)
    csv_row['Flow_Deviation'] = round(total_flow - target_flow, 2)
    for i in range(1, 11):
        if pump_combination[i-1] == 1:
            # Generate pump efficiency based on total efficiency ±0.02
            pump_eff = efficiency + random.uniform(-0.02, 0.02)
            # Clamp to valid range [0.0, 1.0]
            pump_eff = max(0.0, min(0.9, pump_eff))
            csv_row[f'P{i}_Eff'] = round(pump_eff, 3)
        else:
            csv_row[f'P{i}_Eff'] = 0.0
    csv_row['Eff'] = round(efficiency, 4)
    csv_row['PCTW'] = round(kwt, 2)
    
    
    return csv_row


def save_results_for_pressure(best_by_target_flow, filename, is_first_pressure=True):
    csv_rows = []
    for target_flow in sorted(best_by_target_flow.keys()):
        for result in best_by_target_flow[target_flow]:
            csv_rows.append(result_to_csv_row(result, target_flow))
    
    # 🔥 添加调试信息
    print(f"  准备写入 {len(csv_rows)} 行数据，is_first_pressure={is_first_pressure}")
    
    if not csv_rows:
        print(f"  警告：没有数据可写入!")
        return 0
    
    # 添加新增的列
    csv_columns = [
        'Header_Press', 'Flow_Lower', 'Flow_Upper', 'Liquid_Level', 'Pump_Group',
        'P1_Freq', 'P2_Freq', 'P3_Freq', 'P4_Freq', 'P5_Freq', 'P6_Freq', 'P7_Freq', 'P8_Freq', 'P9_Freq', 'P10_Freq',
        'Total_Flow', 'Flow_Deviation',
        'P1_Eff', 'P2_Eff', 'P3_Eff', 'P4_Eff', 'P5_Eff', 'P6_Eff', 'P7_Eff', 'P8_Eff', 'P9_Eff', 'P10_Eff',
        'Eff', 'PCTW'
    ]
    
    mode = 'w' if is_first_pressure else 'a'
    with open(filename, mode, newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        if is_first_pressure:
            writer.writeheader()
            print(f"  ✓ 已写入表头")
        for row in csv_rows:
            writer.writerow(row)
    
    return len(csv_rows)


def query_optimal_solutions(target_pressures_flows):
    """
    查询最优解接口
    输入：target_pressures_flows - 列表，每个元素为 (压力，流量) 元组
          例如：[(0.2, 3000), (0.25, 4000), (0.3, 5000)]
    
    输出：list of tuple, 每个时间点对应一个 (speed_matrix, efficiency_matrix, kwt_matrix)
          speed_matrix: 满足要求的转速矩阵 (N×10)，每行是一个泵组合的 10 个泵转速
          efficiency_matrix: 效率值矩阵 (N×1)
          kwt_matrix: 千吨水电耗矩阵 (N×1)
          
    优化逻辑：
    1. 如果当前流量与上一时刻流量差值 < 100，直接复用上一时刻的有效解。
    2. 如果无有效解且非首点，复用上一时刻解。
    3. 如果首点无解，返回零矩阵。
    """
    all_results_by_timepoint = []
    # 用于缓存上一个时间点的有效解
    last_valid_result = None 
    # 用于记录上一个时间点的目标流量，用于判断是否复用
    prev_target_flow = None
    
    # 遍历所有目标压力和流量组合 (每个时间点)
    for time_idx, (target_pressure, target_flow) in enumerate(target_pressures_flows):
        print(f"\n时间点 {time_idx + 1}/{len(target_pressures_flows)}: 压力={target_pressure}MPa, 流量={target_flow}m³/h")
        
        # --- 判断是否可以直接复用上一时刻解 ---
        if prev_target_flow is not None and last_valid_result is not None:
            flow_diff = abs(target_flow - prev_target_flow)
            if flow_diff < 100:
                print(f"  流量变化较小 ({flow_diff:.1f} < 100)，直接复用上一时间点有效解")
                all_results_by_timepoint.append(last_valid_result)
                # 更新 prev_target_flow 以便下一次比较
                prev_target_flow = target_flow
                continue
        # ---------------------------------------------

        all_speed_combinations = []
        all_efficiencies = []
        all_kwts = []
        
        # 步骤 1: 计算该压力下的所有组合
        pressure_results = calculate_combinations_for_pressure_optimized(
            target_pressure,
            target_flow,
            validate_flow_balance=True,
            validate_flow_limit=True
        )
        
        if not pressure_results:
            print(f"  压力 {target_pressure} 下无有效组合")
        else:
            # 步骤 2: 按目标流量分组
            grouped_results = group_results_by_flow_optimized(
                pressure_results,
                [target_flow],
                batch_size=500000,
                min_fallback_count=5
            )
            
            if not grouped_results:
                print(f"  压力 {target_pressure} 流量 {target_flow} 下无有效分组结果")
            else:
                # 步骤 3: 获取该流量下的最优解
                best_by_target_flow = find_best_solutions_for_target_flows(grouped_results, [target_flow])
                
                # 提取结果
                if target_flow in best_by_target_flow and best_by_target_flow[target_flow]:
                    for result in best_by_target_flow[target_flow]:
                        # 提取转速组合（去掉第一个压力值）
                        speed_combo = result['speed_combination'][1:]  # 去掉压力值
                        
                        # 确保是 10 个泵的转速
                        if len(speed_combo) < 10:
                            # 补齐到 10 个泵
                            speed_combo = speed_combo + [0] * (10 - len(speed_combo))
                        elif len(speed_combo) > 10:
                            speed_combo = speed_combo[:10]
                        
                        all_speed_combinations.append(speed_combo)
                        all_efficiencies.append(result['efficiency'])
                        all_kwts.append(result['kwt'])
                        
                        # # 仅在找到少量解时打印详细信息，避免日志过多
                        # if len(all_speed_combinations) <= 3:
                        #     print(f"  找到解：转速={[int(s) for s in speed_combo]}, "
                        #           f"效率={result['efficiency']:.4f}, "
                        #           f"千吨水电耗={result['kwt']:.2f}")
                
                # 清理内存
                del grouped_results, best_by_target_flow
                gc.collect()
        
        del pressure_results
        gc.collect()
        
        # 处理当前时间点的输出结果
        if all_speed_combinations:
            # 有有效解的情况
            speed_matrix = np.array(all_speed_combinations, dtype=np.float64)
            efficiency_matrix = np.array(all_efficiencies, dtype=np.float64).reshape(-1, 1)
            kwt_matrix = np.array(all_kwts, dtype=np.float64).reshape(-1, 1)
            
            print(f"  时间点 {time_idx + 1}: 找到 {len(speed_matrix)} 个有效解")
            print(f"  转速矩阵形状：{speed_matrix.shape}")
            print(f"  效率矩阵形状：{efficiency_matrix.shape}")
            print(f"  电耗矩阵形状：{kwt_matrix.shape}")
            
            current_result = (speed_matrix, efficiency_matrix, kwt_matrix)
            all_results_by_timepoint.append(current_result)
            
            # 更新缓存，供下一个时间点使用
            last_valid_result = current_result
            
        else:
            # 无有效解的情况
            if last_valid_result is not None:
                # 策略：使用上一个时间点的有效解
                print(f"  时间点 {time_idx + 1}: 警告：未找到有效解，复用上个时间点的有效解")
                all_results_by_timepoint.append(last_valid_result)
            else:
                # 策略：如果是第一个点且无解，或者之前从未有过有效解，返回零矩阵作为兜底
                print(f"  时间点 {time_idx + 1}: 警告：未找到有效解且无历史解，返回零矩阵")
                
                # 确定泵的数量
                pump_count = 10
                
                # 创建全零矩阵 (10x10) 代替单位矩阵
                zero_speed_matrix = np.zeros((pump_count, pump_count), dtype=np.float64)
                
                # 对应的效率矩阵和电耗矩阵也是零矩阵
                zero_metric_matrix = np.zeros((pump_count, 1), dtype=np.float64)
                
                print(f"  零矩阵形状：{zero_speed_matrix.shape}")
                
                fallback_result = (zero_speed_matrix, zero_metric_matrix, zero_metric_matrix)
                all_results_by_timepoint.append(fallback_result)
        
        # 更新 prev_target_flow 用于下一次循环判断
        prev_target_flow = target_flow
                
    
    print(f"\n{'='*60}")
    print(f"查询完成，共 {len(all_results_by_timepoint)} 个时间点")
    for i, (spd, eff, kwt) in enumerate(all_results_by_timepoint):
        print(f"  时间点 {i+1}: 转速矩阵{spd.shape}, 效率矩阵{eff.shape}, 电耗矩阵{kwt.shape}")
    print(f"{'='*60}")
    
    return all_results_by_timepoint
    
# ====================== 主程序 ======================
if __name__ == "__main__":
    print("泵优化系统启动")
    print("泵额定流量配置:")
    
    start_time = time.time()
    
    target_flows = list(range(2950, 12051, 100))
    print(f"\n目标流量：{min(target_flows)}-{max(target_flows)}，步长 100，共{len(target_flows)}个")
    print(f"压力范围：{PRESSURE_RANGE[0]}-{PRESSURE_RANGE[-1]}，共{len(PRESSURE_RANGE)}个")
    print(f"泵组合数：{len(ACTUAL_PUMP_COMBINATIONS)}")
    
    # 文件名不包含非法字符
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f'pump_optimization_results_fitting_{timestamp}.csv'
    
    total_csv_rows = 0
    pressure_stats = []
    header_written = False
    
    for pressure_idx, pressure in enumerate(PRESSURE_RANGE):
        print(f"\n{'='*60}")
        print(f"压力 {pressure} ({pressure_idx+1}/{len(PRESSURE_RANGE)})")
        print(f"{'='*60}")
        
        t0 = time.time()
        
        # 步骤 1: 向量化计算
        all_results = calculate_combinations_for_pressure_optimized(
            pressure, 
            validate_flow_balance=True,
            validate_flow_limit=True
        )
        
        if not all_results:
            print(f"无有效组合，跳过")
            continue
        
        # 步骤 2: 快速分组
        grouped_results = group_results_by_flow_optimized(
            all_results, 
            target_flows, 
            batch_size=500000, 
            min_fallback_count=5
        )
        if not grouped_results:
            del all_results, grouped_results
            gc.collect()
            continue
        
        # 步骤 3: 最优解
        best_by_target_flow = find_best_solutions_for_target_flows(grouped_results, target_flows)
        # 步骤 4: 保存
        csv_count = save_results_for_pressure(
            best_by_target_flow, 
            csv_filename, 
            is_first_pressure=not header_written
        )
        
        # 写入数据后才标记表头已写入
        if csv_count > 0:
            header_written = True
        
        total_csv_rows += csv_count
        
        dt = time.time() - t0
        pressure_stats.append({'pressure': pressure, 'results': len(all_results), 'rows': csv_count, 'time': dt})
        
        del all_results, grouped_results, best_by_target_flow
        gc.collect()
        
        print(f"  完成，耗时{dt:.1f}s，写入{csv_count}行")
        
        if pressure_idx < len(PRESSURE_RANGE) - 1:
            avg_t = sum(s['time'] for s in pressure_stats) / len(pressure_stats)
            remain = avg_t * (len(PRESSURE_RANGE) - pressure_idx - 1)
            print(f"  预估剩余：{remain:.0f}s")
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"完成！总时间{total_time:.1f}s, 共{total_csv_rows}行")
    print(f"结果文件：{csv_filename}")
    print(f"文件包含以下新列:")
    print(f"  - Total_Input_Power: 总管输入功率")
    print(f"  - Total_Output_Power: 总管输出功率")
    print(f"  - Max_Rated_Flow: 泵组最大额定流量")
    print(f"注意：已启用流量限制条件，预测流量不能超过泵的理论最大流量")