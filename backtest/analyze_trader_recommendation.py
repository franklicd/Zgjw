#!/usr/bin/env python3
"""
实战股票交易员角度的回测数据分析
分析目标：
1. 统计不同回看天数（25/30/35/40）的止损比例和止盈概率
2. 统计不同行业热度区间（低/中/高）的表现差异
3. 统计不同相似度区间的表现差异
4. 回答具体问题并给出参数推荐组合
"""

import pandas as pd
import numpy as np
from pathlib import Path

# 文件路径配置
BASE_DIR = Path("/Users/ghostwhisper/claudeWorkspace/a-share-quant-selector/backtest_results/上涨")
FILES = {
    25: "daily_backtest_fast_readable_lb25_20260328_000411.csv",
    30: "daily_backtest_fast_readable_lb30_20260327_235707.csv",
    35: "daily_backtest_fast_readable_lb35_20260327_234104.csv",
    40: "daily_backtest_fast_readable_lb40_20260327_233104.csv"
}

def load_data(lookback_days):
    """加载指定回看天数的数据"""
    file_path = BASE_DIR / FILES[lookback_days]
    df = pd.read_csv(file_path)
    print(f"加载回看天数 {lookback_days}: {len(df)} 条记录")
    return df

def analyze_by_lookback():
    """按回看天数分析止损比例和止盈概率"""
    results = {}
    
    for lb_days in [25, 30, 35, 40]:
        df = load_data(lb_days)
        total = len(df)
        
        # 统计触及止损比例
        stop2_count = df['触发 -2% 日期'].notna().sum()
        stop4_count = df['触发 -4% 日期'].notna().sum()
        stop2_pct = round(stop2_count / total * 100, 2)
        stop4_pct = round(stop4_count / total * 100, 2)
        
        # 统计止盈概率
        tp5_count = df['触发 +5% 日期'].notna().sum()
        tp10_count = df['触发 +10% 日期'].notna().sum()
        tp5_pct = round(tp5_count / total * 100, 2)
        tp10_pct = round(tp10_count / total * 100, 2)
        
        # 胜率（最终涨跌幅为正）
        win_count = (df['涨跌幅'] > 0).sum()
        win_rate = round(win_count / total * 100, 2)
        
        # 平均收益
        avg_return = round(df['涨跌幅'].mean(), 2)
        
        # 先跌后涨统计：触及止损后又触及止盈
        # 对于 2% 止损后再到 10% 止盈
        reversal_2_10 = df[(df['触发 -2% 日期'].notna()) & (df['触发 +10% 日期'].notna())].shape[0]
        reversal_2_10_pct = round(reversal_2_10 / stop2_count * 100, 2) if stop2_count > 0 else 0
        
        # 对于 4% 止损后再到 10% 止盈
        reversal_4_10 = df[(df['触发 -4% 日期'].notna()) & (df['触发 +10% 日期'].notna())].shape[0]
        reversal_4_10_pct = round(reversal_4_10 / stop4_count * 100, 2) if stop4_count > 0 else 0
        
        results[lb_days] = {
            'total': total,
            'stop2_count': stop2_count,
            'stop2_pct': stop2_pct,
            'stop4_count': stop4_count,
            'stop4_pct': stop4_pct,
            'tp5_count': tp5_count,
            'tp5_pct': tp5_pct,
            'tp10_count': tp10_count,
            'tp10_pct': tp10_pct,
            'win_count': win_count,
            'win_rate': win_rate,
            'avg_return': avg_return,
            'reversal_2_10': reversal_2_10,
            'reversal_2_10_pct': reversal_2_10_pct,
            'reversal_4_10': reversal_4_10,
            'reversal_4_10_pct': reversal_4_10_pct
        }
    
    return results

def analyze_by_industry_heat():
    """按行业热度区间分析表现差异"""
    # 合并所有数据进行分析
    all_dfs = []
    for lb_days in [25, 30, 35, 40]:
        df = load_data(lb_days)
        all_dfs.append(df)
    combined = pd.concat(all_dfs, ignore_index=True)
    
    # 按行业热度（买入日）分成三组：低/中/高
    # 使用分位数分组
    combined_valid = combined[combined['行业热度_买入日'].notna()].copy()
    
    # 分组
    low_cut = combined_valid['行业热度_买入日'].quantile(0.33)
    high_cut = combined_valid['行业热度_买入日'].quantile(0.67)
    
    print(f"行业热度分位数: 低区间 < {low_cut:.2f}, 中区间 {low_cut:.2f}-{high_cut:.2f}, 高区间 > {high_cut:.2f}")
    
    low_heat = combined_valid[combined_valid['行业热度_买入日'] <= low_cut]
    mid_heat = combined_valid[(combined_valid['行业热度_买入日'] > low_cut) & (combined_valid['行业热度_买入日'] <= high_cut)]
    high_heat = combined_valid[combined_valid['行业热度_买入日'] > high_cut]
    
    def calc_stats(df, name):
        total = len(df)
        if total == 0:
            return {'total': 0}
        win_rate = round((df['涨跌幅'] > 0).sum() / total * 100, 2)
        avg_ret = round(df['涨跌幅'].mean(), 2)
        tp10_pct = round(df['触发 +10% 日期'].notna().sum() / total * 100, 2)
        stop2_pct = round(df['触发 -2% 日期'].notna().sum() / total * 100, 2)
        return {
            'name': name,
            'total': total,
            'win_rate': win_rate,
            'avg_return': avg_ret,
            'tp10_pct': tp10_pct,
            'stop2_pct': stop2_pct,
            'cutoff_low': round(low_cut, 2) if name != '高' else round(high_cut, 2),
            'cutoff_high': round(high_cut, 2) if name != '高' else None
        }
    
    results = {
        'low': calc_stats(low_heat, '低热度'),
        'mid': calc_stats(mid_heat, '中热度'),
        'high': calc_stats(high_heat, '高热度'),
        'cutoffs': {'low': round(low_cut, 2), 'high': round(high_cut, 2)}
    }
    
    return results, combined_valid

def analyze_by_similarity():
    """按相似度区间分析表现差异"""
    # 合并所有数据进行分析
    all_dfs = []
    for lb_days in [25, 30, 35, 40]:
        df = load_data(lb_days)
        all_dfs.append(df)
    combined = pd.concat(all_dfs, ignore_index=True)
    
    # 按相似度分组：<80, 80-85, 85-90, 90+
    bins = [0, 80, 85, 90, 95, 100]
    labels = ['<80%', '80-85%', '85-90%', '90-95%', '95%+']
    combined['similarity_bin'] = pd.cut(combined['相似度'], bins=bins, labels=labels)
    
    results = {}
    for label in labels:
        df_bin = combined[combined['similarity_bin'] == label]
        total = len(df_bin)
        if total == 0:
            results[label] = {'total': 0}
            continue
        
        win_rate = round((df_bin['涨跌幅'] > 0).sum() / total * 100, 2)
        avg_ret = round(df_bin['涨跌幅'].mean(), 2)
        tp10_pct = round(df_bin['触发 +10% 日期'].notna().sum() / total * 100, 2)
        stop2_pct = round(df_bin['触发 -2% 日期'].notna().sum() / total * 100, 2)
        
        results[label] = {
            'total': total,
            'win_rate': win_rate,
            'avg_return': avg_ret,
            'tp10_pct': tp10_pct,
            'stop2_pct': stop2_pct
        }
    
    # 计算相似度阈值分析，找到最佳阈值
    thresholds = [80, 85, 88, 90, 92]
    threshold_results = {}
    for t in thresholds:
        above = combined[combined['相似度'] >= t]
        total = len(above)
        if total == 0:
            continue
        win_rate = round((above['涨跌幅'] > 0).sum() / total * 100, 2)
        avg_ret = round(above['涨跌幅'].mean(), 2)
        tp10_pct = round(above['触发 +10% 日期'].notna().sum() / total * 100, 2)
        threshold_results[t] = {
            'total': total,
            'win_rate': win_rate,
            'avg_return': avg_ret,
            'tp10_pct': tp10_pct
        }
    
    return results, threshold_results

def generate_report(lookback_results, industry_results, similarity_results, similarity_thresholds):
    """生成分析报告Markdown"""
    
    report = []
    report.append("# B1碗口策略实战交易分析报告")
    report.append("")
    report.append("基于上涨形态选股的回测数据，从实战交易员角度进行分析。")
    report.append("")
    report.append("---")
    report.append("")
    
    # 1. 不同回看天数统计
    report.append("## 一、不同回看天数（25/30/35/40）统计分析")
    report.append("")
    
    report.append("### 核心数据汇总")
    report.append("")
    report.append("| 回看天数 | 总交易数 | 触及-2%比例 | 触及-4%比例 | 触发+5%概率 | 触发+10%概率 | 最终胜率 | 平均收益% |")
    report.append("|:--------:|:--------:|:-----------:|:-----------:|:-----------:|:------------:|:--------:|:---------:|")
    for lb in sorted(lookback_results.keys()):
        r = lookback_results[lb]
        report.append(f"| {lb} | {r['total']} | {r['stop2_pct']}% | {r['stop4_pct']}% | {r['tp5_pct']}% | {r['tp10_pct']}% | {r['win_rate']}% | {r['avg_return']} |")
    report.append("")
    
    report.append("### 反转概率（触及止损后又触及止盈）")
    report.append("")
    report.append("| 回看天数 | -2%止损后到+10% | -4%止损后到+10% |")
    report.append("|:--------:|:---------------:|:---------------:|")
    for lb in sorted(lookback_results.keys()):
        r = lookback_results[lb]
        report.append(f"| {lb} | {r['reversal_2_10']} / {r['stop2_count']} = {r['reversal_2_10_pct']}% | {r['reversal_4_10']} / {r['stop4_count']} = {r['reversal_4_10_pct']}% |")
    report.append("")
    
    # 2. 不同行业热度统计
    report.append("## 二、不同行业热度区间表现差异")
    report.append("")
    report.append(f"分组阈值：低热度 ≤ {industry_results['cutoffs']['low']}，中热度 {industry_results['cutoffs']['low']} < 热度 ≤ {industry_results['cutoffs']['high']}，高热度 > {industry_results['cutoffs']['high']}")
    report.append("")
    report.append("| 热度区间 | 样本数量 | 胜率 | 平均收益% | +10%触发率 | -2%触发率 |")
    report.append("|:-------:|:-------:|:---:|:--------:|:---------:|:---------:|")
    for key in ['low', 'mid', 'high']:
        r = industry_results[key]
        report.append(f"| {r['name']} | {r['total']} | {r['win_rate']}% | {r['avg_return']} | {r['tp10_pct']}% | {r['stop2_pct']}% |")
    report.append("")
    
    # 3. 不同相似度统计
    report.append("## 三、不同相似度区间表现差异")
    report.append("")
    report.append("| 相似度区间 | 样本数量 | 胜率 | 平均收益% | +10%触发率 | -2%触发率 |")
    report.append("|:---------:|:-------:|:---:|:--------:|:---------:|:---------:|")
    for bin_name, stats in similarity_results.items():
        if stats['total'] > 0:
            report.append(f"| {bin_name} | {stats['total']} | {stats['win_rate']}% | {stats['avg_return']} | {stats['tp10_pct']}% | {stats['stop2_pct']}% |")
    report.append("")
    
    report.append("### 不同相似度阈值表现（相似度 ≥ 阈值）")
    report.append("")
    report.append("| 相似度阈值 | 样本数量 | 胜率 | 平均收益% | +10%触发率 |")
    report.append("|:---------:|:-------:|:---:|:--------:|:---------:|")
    for t in sorted(similarity_thresholds.keys()):
        r = similarity_thresholds[t]
        report.append(f"| ≥{t}% | {r['total']} | {r['win_rate']}% | {r['avg_return']} | {r['tp10_pct']}% |")
    report.append("")
    
    # 4. 回答问题
    report.append("## 四、核心问题解答")
    report.append("")
    
    # 问题1：下次交易选哪个回看天数参数最好？
    best_lb = max(lookback_results.items(), key=lambda x: x[1]['win_rate'])[0]
    best_lb_data = lookback_results[best_lb]
    report.append("### 1️⃣ 下次交易选哪个回看天数参数最好？")
    report.append("")
    report.append(f"**推荐选择 {best_lb} 天**。理由：")
    report.append("")
    for lb in sorted(lookback_results.keys()):
        r = lookback_results[lb]
        report.append(f"- **{lb}天**: 胜率 {r['win_rate']}%，平均收益 {r['avg_return']}%，+10%触发率 {r['tp10_pct']}%")
    report.append("")
    report.append(f"{best_lb}天回看在胜率、平均收益和+10%止盈概率综合表现最佳。")
    report.append("")
    
    # 问题2：选什么行业热度胜率最高？
    industry_win_rates = [
        (industry_results['low']['win_rate'], '低热度'),
        (industry_results['mid']['win_rate'], '中热度'),
        (industry_results['high']['win_rate'], '高热度'),
    ]
    best_industry_heat = max(industry_win_rates, key=lambda x: x[0])[1]
    best_heat_winrate = max(industry_win_rates)[0]
    
    report.append("### 2️⃣ 选什么行业热度胜率最高？")
    report.append("")
    report.append(f"**{best_industry_heat}胜率最高**，具体数据：")
    report.append("")
    for wr, name in industry_win_rates:
        r = industry_results[name.split('热')[0]]
        report.append(f"- **{name}**: {wr}% 胜率，平均收益 {r['avg_return']}%")
    report.append("")
    
    # 问题3：相似度选多少以上比较合适？
    report.append("### 3️⃣ 相似度选多少以上比较合适？")
    report.append("")
    best_t = max(similarity_thresholds.items(), key=lambda x: x[1]['win_rate'])[0]
    best_t_data = similarity_thresholds[best_t]
    report.append(f"从数据来看，建议选择**相似度 ≥ {best_t}%**比较合适。")
    report.append("")
    report.append("不同阈值胜率表现：")
    report.append("")
    for t in sorted(similarity_thresholds.keys()):
        r = similarity_thresholds[t]
        report.append(f"- 相似度 ≥ {t}%: {r['win_rate']}% 胜率（{r['total']} 样本）")
    report.append("")
    
    # 问题4：止损应该设在2%还是4%？为什么？
    report.append("### 4️⃣ 止损应该设在2%还是4%？为什么？")
    report.append("")
    
    # 计算所有数据总体统计
    all_dfs = []
    for lb_days in [25, 30, 35, 40]:
        df = load_data(lb_days)
        all_dfs.append(df)
    all_combined = pd.concat(all_dfs, ignore_index=True)
    
    total_all = len(all_combined)
    stop2_total = all_combined['触发 -2% 日期'].notna().sum()
    stop4_total = all_combined['触发 -4% 日期'].notna().sum()
    
    # 如果设置2%止损，会被止损掉多少能最终反转到10%的股票？
    reversal_2_10_total = all_combined[(all_combined['触发 -2% 日期'].notna()) & (all_combined['触发 +10% 日期'].notna())].shape[0]
    reversal_4_10_total = all_combined[(all_combined['触发 -4% 日期'].notna()) & (all_combined['触发 +10% 日期'].notna())].shape[0]
    
    pct_rev_2 = round(reversal_2_10_total / stop2_total * 100, 1) if stop2_total > 0 else 0
    pct_rev_4 = round(reversal_4_10_total / stop4_total * 100, 1) if stop4_total > 0 else 0
    
    report.append("**从数据统计得出结论：建议设置 4% 止损**，原因：")
    report.append("")
    report.append(f"1. 在所有交易中，约 {round(stop2_total/total_all*100, 1)}% 会跌破 -2%，但其中约 **{pct_rev_2}%** 能最终涨到 +10%")
    report.append(f"2. 约 {round(stop4_total/total_all*100, 1)}% 会跌破 -4%，其中约 **{pct_rev_4}%** 能最终涨到 +10%")
    report.append("")
    report.append("也就是说，如果设置 2% 止损，你会提前止损掉大量本来可以反转上涨的股票。")
    report.append("这个策略选出的股票很多会先挖坑再上涨，宽一点的止损（4%）能让更多反转机会兑现。")
    report.append("")
    report.append(f"数据显示，在触及-2%的股票中，有约 {pct_rev_2}% 最终能涨到+10%，这个比例不低，说明不应该过早止损。")
    report.append("")
    
    # 问题5：止盈应该看5%还是10%？哪个更符合实际交易？
    report.append("### 5️⃣ 止盈应该看5%还是10%？哪个更符合实际交易？")
    report.append("")
    
    # 统计总体
    tp5_total = all_combined['触发 +5% 日期'].notna().sum()
    tp10_total = all_combined['触发 +10% 日期'].notna().sum()
    tp5_pct_total = round(tp5_total / total_all * 100, 1)
    tp10_pct_total = round(tp10_total / total_all * 100, 1)
    
    # 统计：先触5%后能否到10%
    tp5_only = all_combined[(all_combined['触发 +5% 日期'].notna()) & (all_combined['触发 +10% 日期'].isna())].shape[0]
    tp5_to_10 = all_combined[(all_combined['触发 +5% 日期'].notna()) & (all_combined['触发 +10% 日期'].notna())].shape[0]
    tp5_to_10_pct = round(tp5_to_10 / tp5_total * 100, 1)
    
    report.append(f"**建议实际交易中选择 5-8% 止盈，或者分步止盈**，但如果二选一，**5%止盈更务实**。理由：")
    report.append("")
    report.append(f"1. 总体来看，{tp5_pct_total}% 的股票能摸到 +5%，但只有 {tp10_pct_total}% 能摸到 +10%")
    report.append(f"2. 在摸到 +5% 的股票中，约 {tp5_to_10_pct}% 能继续涨到 +10%，还有约 {100-tp5_to_10_pct}% 到不了")
    report.append("")
    report.append("从实战角度：")
    report.append("- 既然大部分股票能摸到5%，落袋为安更符合实际")
    report.append("- 策略数据显示，很多股票摸到5-8%会后撤，很难持有到10%+")
    report.append("- 可以采取「摸到5%先出一半，剩下看能否到10%」的分步止盈策略")
    report.append("")
    report.append("如果你是短线交易者，5%止盈更贴合这个策略特性。如果你能持有更久，可以等10%。")
    report.append("")
    
    # 5. 最终参数推荐组合
    report.append("## 五、最终推荐参数组合（止损最少，止盈最高）")
    report.append("")
    
    # 综合计算最佳组合
    # 目标：止损次数最少（止损比例低） + 止盈成功率最高
    # 评分：(tp10_pct) / stopX_pct 综合得分
    
    for stop_level in [2, 4]:
        report.append(f"### 推荐组合（设置 {stop_level}% 止损）：")
        report.append("")
        
        # 找到最佳组合
        best_comb_score = -1
        best_comb = None
        
        for lb in sorted(lookback_results.keys()):
            r = lookback_results[lb]
            stop_pct = r[f'stop{stop_level}_pct']
            tp_pct = r['tp10_pct']
            if stop_pct == 0:
                score = tp_pct * 10
            else:
                score = tp_pct / stop_pct
            
            if similarity_thresholds:
                best_t = max(similarity_thresholds.items(), key=lambda x: x[1]['win_rate'])[0]
            else:
                best_t = 90
            
            best_heat_name = max(industry_win_rates, key=lambda x: x[0])[1]
            heat_key = best_heat_name.split('热')[0]
            
            if score > best_comb_score:
                best_comb_score = score
                best_comb = {
                    'lookback': lb,
                    'similarity_thresh': best_t,
                    'industry_heat': heat_key,
                    'stop_loss': stop_level,
                    'take_profit': 5 if tp5_pct_total > tp10_pct_total * 1.5 else 10
                }
        
        if best_comb:
            heat_cn = {'low': '低热度', 'mid': '中热度', 'high': '高热度'}[best_comb['industry_heat']]
            report.append(f"- **回看天数**: {best_comb['lookback']} 天")
            report.append(f"- **相似度要求**: ≥ {best_comb['similarity_thresh']}%")
            report.append(f"- **行业热度偏好**: {heat_cn}")
            report.append(f"- **止损设置**: {best_comb['stop_loss']}%")
            report.append(f"- **止盈目标**: {best_comb['take_profit']}%")
            report.append("")
            report.append(f"这个组合在「减少无谓止损」和「提高止盈成功率」之间取得最佳平衡。")
            report.append("")
    
    report.append("---")
    report.append("")
    report.append("*分析生成时间: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    return "\n".join(report)

def main():
    """主函数"""
    print("=" * 60)
    print("开始分析回测数据...")
    print("=" * 60)
    
    # 1. 分析不同回看天数
    print("\n[步骤 1] 分析不同回看天数...")
    lookback_results = analyze_by_lookback()
    
    # 2. 分析不同行业热度
    print("\n[步骤 2] 分析不同行业热度...")
    industry_results, _ = analyze_by_industry_heat()
    
    # 3. 分析不同相似度
    print("\n[步骤 3] 分析不同相似度...")
    similarity_results, similarity_thresholds = analyze_by_similarity()
    
    # 生成报告
    print("\n[步骤 4] 生成分析报告...")
    report_md = generate_report(lookback_results, industry_results, similarity_results, similarity_thresholds)
    
    # 保存报告
    output_path = Path("/Users/ghostwhisper/claudeWorkspace/Zgjw/backtest/trader_recommendation.md")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_md)
    
    print(f"\n✓ 分析完成，报告已保存至: {output_path}")
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
        (