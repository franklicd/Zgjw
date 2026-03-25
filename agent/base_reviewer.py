"""
base_reviewer.py
~~~~~~~~~~~~~~~~
提供 LLM 图表分析的基础架构（支持单线程/多线程模式）：
- 加载配置和 prompt
- 读取候选股票列表
- 查找本地 K 线图
- 支持单线程顺序调用或线程池并发调用
- 结果汇总和输出
- 多线程进度实时追踪（带进度条和统计）
- 行业数据分析（所属行业、行业热度）
"""

import json
import re
import time
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from .industry_analyzer import IndustryAnalyzer
except ImportError:
    from industry_analyzer import IndustryAnalyzer


class ProgressTracker:
    """多线程进度追踪器"""
    def __init__(self, total: int):
        self.total = total
        self.completed = 0
        self.failed = 0
        self.success = 0
        self.lock = threading.Lock()
        self.start_time = time.time()

    def add_result(self, success: bool):
        with self.lock:
            self.completed += 1
            if success:
                self.success += 1
            else:
                self.failed += 1

    def get_progress(self) -> tuple[int, int, int, int]:
        with self.lock:
            return self.completed, self.success, self.failed, self.total - self.completed

    def get_elapsed(self) -> float:
        return time.time() - self.start_time

    def get_eta(self) -> float:
        elapsed = self.get_elapsed()
        completed = self.completed
        if completed == 0:
            return 0
        avg_time = elapsed / completed
        remaining = self.total - completed
        return avg_time * remaining


class BaseReviewer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.prompt = self.load_prompt(Path(config["prompt_path"]))
        self.kline_dir = Path(config["kline_dir"])
        self.output_dir = Path(config["output_dir"])
        self.data_dir = Path(config.get("data_dir", Path(__file__).parent.parent / "data"))

        # 初始化行业分析器
        self.industry_analyzer = IndustryAnalyzer(data_dir=self.data_dir / "industry")
        self._load_industry_data()
        self._industry_heat: Dict = {}  # 行业热度缓存

    def _load_industry_data(self):
        """加载行业数据（从 stocklist.csv）"""
        # 尝试从配置文件或默认路径加载 stocklist
        stocklist_paths = [
            Path(self.config["stocklist"]) if self.config.get("stocklist") else None,  # 配置中的路径
            Path(__file__).parent.parent / "pipeline" / "stocklist.csv",  # 默认路径
        ]
        for stocklist_path in stocklist_paths:
            if stocklist_path and stocklist_path.exists() and stocklist_path.is_file():
                self.industry_analyzer.load_industry_from_stocklist(stocklist_path)
                print(f"[INFO] 已加载行业信息：{len(self.industry_analyzer._industry_cache)} 只股票")
                break

    @staticmethod
    def load_prompt(prompt_path: Path) -> str:
        return prompt_path.read_text(encoding="utf-8")

    @staticmethod
    def load_candidates(path: Path) -> dict:
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def find_chart_images(self, pick_date: str, code: str) -> Optional[Path]:
        date_dir = self.kline_dir / pick_date
        day_chart = date_dir / f"{code}_day.jpg"
        if not day_chart.exists():
            day_chart_png = date_dir / f"{code}_day.png"
            day_chart = day_chart_png if day_chart_png.exists() else None
        return day_chart

    @staticmethod
    def extract_json(text: str) -> dict:
        code_block = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if code_block:
            text = code_block.group(1)
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError(f"未能在模型输出中找到 JSON 对象:\n{text}")
        return json.loads(text[start:end])

    def review_stock(self, code: str, day_chart: Path, prompt: str) -> dict:
        """子类需实现此方法，调用具体的 LLM 进行打分，并返回 JSON 解析字典。"""
        raise NotImplementedError("子类必须实现 review_stock 方法")

    def generate_suggestion(self, pick_date: str, all_results: List[dict], min_score: float) -> dict:
        """
        生成推荐建议，包含行业信息和行业热度

        Args:
            pick_date: 选股日期
            all_results: 所有股票的评分结果
            min_score: 最低分数门槛

        Returns:
            推荐建议字典，包含行业信息
        """
        passed = [r for r in all_results if r.get("total_score", 0) >= min_score]
        excluded = [r["code"] for r in all_results if r.get("total_score", 0) < min_score]

        passed.sort(key=lambda r: r.get("total_score", 0), reverse=True)

        # 获取行业热度数据（如果还没有加载）
        if not self._industry_heat:
            print("[INFO] 正在计算行业热度...")
            self._industry_heat = self.industry_analyzer.calculate_industry_heat()
            if self._industry_heat:
                print(f"       共 {len(self._industry_heat)} 个行业")

        recommendations = []
        for i, r in enumerate(passed):
            code = r["code"]
            industry = self.industry_analyzer.get_stock_industry(code)
            industry_data = self._industry_heat.get(industry, {}) if industry else {}

            rec = {
                "rank": i + 1,
                "code": r["code"],
                "verdict": r.get("verdict", ""),
                "total_score": r.get("total_score", 0),
                "signal_type": r.get("signal_type", ""),
                "comment": r.get("comment", ""),
                "industry": industry or "未知",
                "industry_turnover": industry_data.get('turnover', 0),  # 行业成交额（亿元）
                "industry_market_ratio": industry_data.get('market_ratio', 0),  # 行业占比（%）
                "industry_change_pct": industry_data.get('change_pct', 0),  # 行业涨跌幅
            }
            recommendations.append(rec)

        return {
            "date": pick_date,
            "min_score_threshold": min_score,
            "total_reviewed": len(all_results),
            "recommendations": recommendations,
            "excluded": excluded,
            "industry_heat": self._industry_heat,  # 附加行业热度数据
        }

    def _process_single_stock(self, candidate: dict, pick_date: str, out_dir: Path) -> tuple[Optional[dict], Optional[str]]:
        """处理单只股票的分析，单线程顺序调用"""
        code: str = candidate["code"]
        out_file = out_dir / f"{code}.json"

        # 跳过已存在的结果
        if self.config.get("skip_existing", False) and out_file.exists():
            try:
                with open(out_file, encoding="utf-8") as f:
                    result = json.load(f)
                return result, None
            except json.JSONDecodeError as e:
                print(f"[⚠️ ] {code} — 缓存文件损坏，重新分析：{e}")
                out_file.unlink(missing_ok=True)

        # 检查图表是否存在
        day_chart = self.find_chart_images(pick_date, code)
        if day_chart is None:
            return None, code

        try:
            print(f"  → 正在分析 {code}...")
            result = self.review_stock(
                code=code,
                day_chart=day_chart,
                prompt=self.prompt,
            )
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)


            return result, None
        except Exception as e:
            print(f"[❌] {code} — 分析失败：{e}")
            return None, code

    def run(self):
        candidates_data = self.load_candidates(Path(self.config["candidates"]))
        pick_date: str = candidates_data["pick_date"]
        candidates: List[dict] = candidates_data["candidates"]
        print(f"[INFO] pick_date={pick_date}，候选股票数={len(candidates)}")

        out_dir = self.output_dir / pick_date
        out_dir.mkdir(parents=True, exist_ok=True)

        all_results: List[dict] = []
        failed_codes: List[str] = []


        # 先处理已经存在的缓存文件
        cached_count = 0
        for candidate in candidates:
            code = candidate["code"]
            out_file = out_dir / f"{code}.json"
            if self.config.get("skip_existing", False) and out_file.exists():
                try:
                    with open(out_file, encoding="utf-8") as f:
                        result = json.load(f)
                    all_results.append(result)
                    cached_count += 1
                    print(f"[✅] {code} — 已缓存，跳过")
                except:
                    pass

        # 过滤出需要处理的股票
        to_process = [c for c in candidates if not (self.config.get("skip_existing", False) and (out_dir / f"{c['code']}.json").exists())]
        print(f"[INFO] 待分析股票数：{len(to_process)} (已缓存 {cached_count} 支)")

        if to_process:
            max_workers = self.config.get("max_workers", 1)
            request_delay = self.config.get("request_delay", 0.2)  # 默认 0.2 秒

            if max_workers <= 1:
                # 单线程模式
                print(f"[INFO] 单线程模式已启用，顺序调用 AI API（请求间隔：{request_delay}秒）")
                total = len(to_process)
                for i, candidate in enumerate(to_process, 1):
                    code = candidate["code"]
                    # 应用请求间隔
                    if request_delay > 0:
                        time.sleep(request_delay)
                    # 显示当前进度（开始处理前也输出）
                    print(f"\n[{i}/{total}] 正在处理：{code}")
                    result, failed_code = self._process_single_stock(candidate, pick_date, out_dir)
                    if result:
                        all_results.append(result)
                        verdict = result.get("verdict", "?")
                        score = result.get("total_score", "?")
                        print(f"[✅] {code} — 完成 | verdict={verdict}, score={score}")
                    if failed_code:
                        failed_codes.append(failed_code)
            else:
                # 多线程并发模式
                print(f"[INFO] 多线程并发模式已启用，最大并发数：{max_workers}")
                print(f"       请求间隔：{request_delay}秒")
                print(f"       候选总数：{len(to_process)}\n")

                # 初始化进度追踪器
                tracker = ProgressTracker(len(to_process))

                # 应用请求间隔
                request_delay = self.config.get("request_delay", 0)

                def process_with_delay(candidate):
                    code = candidate["code"]
                    if request_delay > 0:
                        time.sleep(request_delay)
                    result, failed_code = self._process_single_stock(candidate, pick_date, out_dir)
                    success = result is not None
                    tracker.add_result(success)

                    # 实时进度输出
                    completed, success_cnt, failed_cnt, pending = tracker.get_progress()
                    elapsed = tracker.get_elapsed()
                    eta = tracker.get_eta()
                    pct = (completed / tracker.total) * 100

                    status = "✅" if success else "❌"
                    score_info = ""
                    if result:
                        verdict = result.get("verdict", "?")
                        score = result.get("total_score", "?")
                        score_info = f"| verdict={verdict}, score={score}"

                    print(f"\r[{completed}/{tracker.total}] {status} {code} {score_info}  "
                          f"({pct:.1f}% | 成功:{success_cnt} 失败:{failed_cnt} | 剩余:{pending} | "
                          f"耗时:{elapsed:.0f}s | 预计:{eta:.0f}s)", end="", flush=True)

                    return result, failed_code

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {executor.submit(process_with_delay, c): c for c in to_process}

                    for future in as_completed(futures):
                        try:
                            result, failed_code = future.result()
                            if result:
                                all_results.append(result)
                            if failed_code:
                                failed_codes.append(failed_code)

                        except Exception as e:
                            code = futures[future]["code"]
                            print(f"\n[❌] {code} — 异常：{e}")
                            failed_codes.append(code)

                # 换行
                print()

        print(f"\n[INFO] 评分完成：成功 {len(all_results)} 支，失败/跳过 {len(failed_codes)} 支")
        if failed_codes:
            print(f"[WARN] 未处理股票：{failed_codes}")

        if not all_results:
            print("[ERROR] 没有可用的评分结果，跳过汇总。")
            return

        print("\n[INFO] 正在生成汇总推荐建议 ...")
        min_score = self.config.get("suggest_min_score", 4.0)
        suggestion = self.generate_suggestion(
            pick_date=pick_date,
            all_results=all_results,
            min_score=min_score,
        )
        suggestion_file = out_dir / "suggestion.json"
        with open(suggestion_file, "w", encoding="utf-8") as f:
            json.dump(suggestion, f, ensure_ascii=False, indent=2)
        print(f"[INFO] 汇总推荐已写入：{suggestion_file}")
        print(f"       推荐股票数（score≥{min_score}）: {len(suggestion['recommendations'])}")

        print("\n✅ 全部完成。")
        print(f"   输出目录：{out_dir}")
