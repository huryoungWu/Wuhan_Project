import numpy as np
import sys
import os
import json
import __init__

from long_term_inference import long_term_predict
from Find_Strategy.brute_search_mianyang import query_optimal_solutions

# ================= 配置区域 =================
PENALTY_STOP_START = 1000
PENALTY_FREQ_COEF = 0.5
PUMP_NUMB = 10
PENALTY_OVERTIME = 100000
TOP_K = 10  # 每个状态保留 TOP-K 条路径
# ===========================================

def generate_daily_pump_schedule(schedule, time_interval_minutes=30, start_hour=0, start_minute=0):
    """
    从调度结果生成每日泵组合安排，并格式化输出
    
    参数:
    - schedule: 调度结果列表
    - time_interval_minutes: 时间间隔（分钟），默认30分钟
    - start_hour: 起始小时，默认0
    - start_minute: 起始分钟，默认0
    
    返回:
    - daily_combinations: 按时间段合并的泵组合列表
    """
    
    if not schedule:
        print("⚠️ 无调度安排数据")
        return []
    
    print("\n" + "="*60)
    print("每日泵组合安排")
    print("="*60)
    # 1. 计算每个时间点对应的实际时间 (作为该步长的起始时间)
    time_points = []
    for i, entry in enumerate(schedule):
        # 计算从起始时间开始经过的分钟数
        total_minutes = i * time_interval_minutes
        hours_from_start = total_minutes // 60
        minutes_from_start = total_minutes % 60
        
        # 计算实际时间
        current_hour = (start_hour + hours_from_start) % 24
        current_minute = (start_minute + minutes_from_start) % 60
        
        # 处理分钟进位
        if start_minute + minutes_from_start >= 60:
            current_hour = (current_hour + 1) % 24
        
        # 格式化时间为 HH:MM
        time_str = f"{current_hour:02d}:{current_minute:02d}"
        time_points.append({
            "time": time_str,
            "pump_combination": entry["pump_combination"],
            "group_id": entry["group_id"],
            "active_pump_count": entry["active_pump_count"],
            "original_index": i
        })
    
    # 2. 合并连续的相同泵组合
    daily_combinations = []
    
    if not time_points:
        return daily_combinations
    
    # 起始第一个时间段
    start_time = time_points[0]["time"]
    start_index = 0 # 0-based index in time_points
    current_combination = time_points[0]["pump_combination"]
    current_group_id = time_points[0]["group_id"]
    
    for i in range(1, len(time_points)):
        # 检查泵组合是否相同
        if time_points[i]["pump_combination"] == current_combination:
            continue
        else:
            # 泵组合变化，记录上一个时间段
            # 上一个时间段包含 time_points[start_index] 到 time_points[i-1]
            # 它的起始时间是 time_points[start_index]['time']
            # 它的结束时间应该是 time_points[i]['time'] (即下一个状态开始的时间)
            
            end_time = time_points[i]["time"]
            
            # 计算持续步长数
            num_steps = i - start_index
            duration_minutes = num_steps * time_interval_minutes
            
            # 获取开启的泵编号
            on_pumps = [j+1 for j, status in enumerate(current_combination) if status == 1]
            
            combination_info = {
                "time_range": f"{start_time}-{end_time}",
                "start_time": start_time,
                "end_time": end_time,
                "start_index": start_index + 1,  # 转换为1-based索引，方便人类阅读
                "end_index": i,                  # 1-based index of the last step in this block? Or the next start?
                                                 # 通常 end_index 指向包含的最后一个元素的索引(1-based)
                "pump_combination": current_combination,
                "group_id": current_group_id,
                "active_pump_count": time_points[i-1]["active_pump_count"],
                "duration_minutes": duration_minutes,
                "on_pumps": on_pumps,
                "on_pumps_str": ', '.join([f'P{p}' for p in on_pumps]) if on_pumps else '无'
            }
            
            daily_combinations.append(combination_info)
            
            # 开始新的时间段
            start_time = time_points[i]["time"]
            start_index = i
            current_combination = time_points[i]["pump_combination"]
            current_group_id = time_points[i]["group_id"]
    
    
    last_point_time = time_points[-1]["time"]
    
    # 计算最后一个时间点的结束时间
    total_minutes_last = (len(time_points) - 1) * time_interval_minutes + time_interval_minutes
    hours_end = (start_hour + total_minutes_last // 60) % 24
    mins_end = (start_minute + total_minutes_last % 60) % 60
    if start_minute + total_minutes_last % 60 >= 60:
        hours_end = (hours_end + 1) % 24
        
    end_time_final = f"{hours_end:02d}:{mins_end:02d}"
    
    num_steps_last = len(time_points) - start_index
    duration_minutes_last = num_steps_last * time_interval_minutes
    
    # 获取开启的泵编号
    on_pumps = [j+1 for j, status in enumerate(current_combination) if status == 1]
    
    combination_info = {
        "time_range": f"{start_time}-{end_time_final}",
        "start_time": start_time,
        "end_time": end_time_final,
        "start_index": start_index + 1,
        "end_index": len(time_points), # 1-based index of the last element
        "pump_combination": current_combination,
        "group_id": current_group_id,
        "active_pump_count": time_points[-1]["active_pump_count"],
        "duration_minutes": duration_minutes_last,
        "on_pumps": on_pumps,
        "on_pumps_str": ', '.join([f'P{p}' for p in on_pumps]) if on_pumps else '无'
    }
    
    daily_combinations.append(combination_info)
    
    return daily_combinations

def _print_detailed_schedule(daily_combinations):
    """打印详细调度信息"""
    if not daily_combinations:
        print("无调度安排")
        return
    
    print(f"总时间段数: {len(daily_combinations)}")
    print(f"时间间隔: {daily_combinations[0]['duration_minutes'] // (daily_combinations[0]['end_index'] - daily_combinations[0]['start_index'] + 1) if len(daily_combinations) > 0 else 0}分钟")
    print()
    
    for i, item in enumerate(daily_combinations, 1):
        print(f"{i}. 时间段: {item['time_range']}")
        print(f"   时间点范围: T{item['start_index']}-T{item['end_index']}")
        print(f"   持续时长: {item['duration_minutes']}分钟")
        print(f"   泵组ID: {item['group_id']}")
        print(f"   运行泵数量: {item['active_pump_count']}台")
        print(f"   开启的泵: {item['on_pumps_str']}")
        print(f"   泵状态组合: {item['pump_combination']}")
        print()
def _custom_penalty_score(x, y):
    base_score = x + y
    if (x == 0 and y != 0) or (x != 0 and y == 0):
        return base_score - PENALTY_STOP_START
    if x == 0 and y == 0:
        return 0
    if x != y:
        return base_score - (PENALTY_FREQ_COEF * abs(x - y))
    return base_score


def _calculate_transition_matrix(M_prev, M_curr):
    num_groups_prev = M_prev.shape[0]
    num_groups_curr = M_curr.shape[0]
    num_pumps = M_prev.shape[1]

    trans_matrix = np.zeros((num_groups_prev, num_groups_curr))

    for i in range(num_groups_prev):
        for j in range(num_groups_curr):
            total = 0
            for k in range(num_pumps):
                total += _custom_penalty_score(M_prev[i][k], M_curr[j][k])
            trans_matrix[i][j] = total

    return trans_matrix


# ------------------------------------------------------------------------------
# 🔥 全局最优 DP 核心：每个状态 [group] 保留 TOP-K 条最优路径
# ------------------------------------------------------------------------------
def get_optimal_pump_schedule(
    results_list,
    initial_pump_status=None,
    initial_running_hours=None,
    time_step_hours=1.0,
    max_continuous_hours=24.0
):
    time_matrices = [res[0] for res in results_list]
    efficiency_matrices = [res[1] for res in results_list]
    kwt_matrices = [res[2] for res in results_list]

    if not time_matrices:
        raise ValueError("无有效时间点数据")

    N = len(time_matrices)
    n_pumps = time_matrices[0].shape[1]

    # ================= DP 结构 =================
    # dp[t][j] = 列表，保存 TOP-K 条最优路径信息：
    # { "score": float, "parent_group": int, "run_hours": array }
    dp = []  

    # ================= t=0 初始化 =================
    M0 = time_matrices[0]
    n_groups0 = M0.shape[0]
    dp0 = [[] for _ in range(n_groups0)]

    if initial_pump_status is not None:
        init_arr = np.array(initial_pump_status).reshape(1, -1)
        trans_init = _calculate_transition_matrix(init_arr, M0)
        for j in range(n_groups0):
            score = trans_init[0][j]
            rh = np.zeros(n_pumps)
            for p in range(n_pumps):
                if M0[j][p] > 0 and initial_pump_status[p] > 0:
                    rh[p] = initial_running_hours[p] + time_step_hours
                elif M0[j][p] > 0:
                    rh[p] = time_step_hours
                else:
                    rh[p] = 0
            dp0[j].append({"score": score, "parent_group": -1, "run_hours": rh})
    else:
        for j in range(n_groups0):
            score = np.sum(M0[j])
            rh = np.where(M0[j] > 0, time_step_hours, 0)
            dp0[j].append({"score": score, "parent_group": -1, "run_hours": rh})

    dp.append(dp0)

    # ================= DP 递推：保留 TOP-K 路径 =================
    for t in range(1, N):
        M_prev = time_matrices[t-1]
        M_curr = time_matrices[t]
        n_prev = M_prev.shape[0]
        n_curr = M_curr.shape[0]

        trans_mat = _calculate_transition_matrix(M_prev, M_curr)
        dp_t = [[] for _ in range(n_curr)]

        for j in range(n_curr):
            candidates = []
            # 遍历上一步所有 group
            for i in range(n_prev):
                # 遍历上一步该 group 保留的 TOP-K 路径
                for path_prev in dp[t-1][i]:
                    score_prev = path_prev["score"]
                    rh_prev = path_prev["run_hours"]
                    trans_score = trans_mat[i][j]
                    new_score = score_prev + trans_score

                    # 计算新运行时间 & 合法性
                    new_rh = rh_prev.copy()
                    valid = True
                    for p in range(n_pumps):
                        prev_on = M_prev[i][p] > 0
                        curr_on = M_curr[j][p] > 0
                        if curr_on:
                            if prev_on:
                                new_rh[p] += time_step_hours
                            else:
                                new_rh[p] = time_step_hours
                            if new_rh[p] > max_continuous_hours:
                                valid = False
                                break
                        else:
                            new_rh[p] = 0
                    if not valid:
                        continue

                    candidates.append({
                        "score": new_score,
                        "parent_group": i,
                        "run_hours": new_rh
                    })

            # 按得分从高到低排序，保留 TOP-K 条最优路径
            candidates = sorted(candidates, key=lambda x: -x["score"])
            dp_t[j] = candidates[:TOP_K]

        dp.append(dp_t)

    # ================= 回溯：从所有时刻、所有路径中找全局最高分 =================
    best_final_score = -np.inf
    best_final_state = None

    last_t = N - 1
    last_groups = time_matrices[last_t].shape[0]

    for j in range(last_groups):
        for path_idx, path in enumerate(dp[last_t][j]):
            if path["score"] > best_final_score:
                best_final_score = path["score"]
                best_final_state = {
                    "t": last_t,
                    "group": j,
                    "path_idx": path_idx
                }

    if best_final_state is None:
        raise RuntimeError("无可行解")

    # ================= 回溯整条路径 =================
    path = []
    curr = best_final_state
    while True:
        t = curr["t"]
        j = curr["group"]
        path_idx = curr["path_idx"]
        path.append((t, j))

        parent_g = dp[t][j][path_idx]["parent_group"]
        if parent_g == -1:
            break

        # 找到父节点对应的路径索引
        best_parent_score = dp[t][j][path_idx]["score"] - _calculate_transition_matrix(
            time_matrices[t-1], time_matrices[t]
        )[parent_g][j]

        parent_path_idx = 0
        for idx, p_path in enumerate(dp[t-1][parent_g]):
            if abs(p_path["score"] - best_parent_score) < 1e-6:
                parent_path_idx = idx
                break

        curr = {
            "t": t - 1,
            "group": parent_g,
            "path_idx": parent_path_idx
        }

    path.reverse()

    # ================= 输出结果（格式完全和你原来一样） =================
    schedule = []
    for idx, (t, g) in enumerate(path):
        speeds = time_matrices[t][g].tolist()
        group_eff = efficiency_matrices[t][g]
        group_kwt = kwt_matrices[t][g]

        group_eff = float(group_eff[0]) if hasattr(group_eff, "__len__") else float(group_eff)
        group_kwt = float(group_kwt[0]) if hasattr(group_kwt, "__len__") else float(group_kwt)

        rh = dp[t][g][0]["run_hours"].round(2).tolist()
        speeds = [float(v) for v in speeds]
        pump_comb = [1 if s > 0 else 0 for s in speeds]

        schedule.append({
            "time": f"T{t+1}",
            "time_index": t + 1,
            "group_id": int(g + 1),
            "pump_status": speeds,
            "pump_combination": pump_comb,
            "group_efficiency": group_eff,
            "group_kwt": group_kwt,
            "active_pump_count": sum(pump_comb),
            "running_hours": rh
        })

    return schedule


# ================= 主程序 =================
if __name__ == "__main__":

    INITIAL_PUMP_STATE = [1450, 1450, 0, 0, 0, 0, 0, 0, 0, 0]
    INITIAL_RUNNING_HOURS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    TIME_STEP_HOURS = 0.5
    MAX_CONTINUOUS_HOURS = 24.0
    TIME_INTERVAL_MINUTES = int(TIME_STEP_HOURS * 60)

    if len(INITIAL_PUMP_STATE) != PUMP_NUMB or len(INITIAL_RUNNING_HOURS) != PUMP_NUMB:
        exit("初始状态长度错误！")

    csv_path = r"Long_Term_Flow_Predict_And_Find_Strategy\results_seq2seq\inference_input_best_case.csv"
    targets = long_term_predict(csv_path)
    # targets = [(0.20, 3000), (0.20, 3000), (0.40, 6000), (0.40, 7000)]

    print("=" * 60)
    print("开始优化调度")
    print("=" * 60)

    results_list = query_optimal_solutions(targets)

    try:
        schedule = get_optimal_pump_schedule(
            results_list,
            initial_pump_status=INITIAL_PUMP_STATE,
            initial_running_hours=INITIAL_RUNNING_HOURS,
            time_step_hours=TIME_STEP_HOURS,
            max_continuous_hours=MAX_CONTINUOUS_HOURS
        )

        print("\n✅ 最优调度计划：")
        daily_combinations = generate_daily_pump_schedule(schedule,TIME_INTERVAL_MINUTES,0,0)
        print(daily_combinations)

        _print_detailed_schedule(daily_combinations)
        

    except Exception as e:
        print(f"\n错误：{e}")
        import traceback
        traceback.print_exc()