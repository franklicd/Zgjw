"""
base_reviewer.py
~~~~~~~~~~~~~~~~
提供 LLM 图表分析的基础架构（支持多线程并发）：
- 加载配置和 prompt
- 读取候选股票列表
- 查找本地 K 线图
- 多线程并发调用子类实现的单股评分模型
- 结果汇总和输出
"""

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial


class BaseReviewer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.prompt = self.load_prompt(Path(config["prompt_path"]))
        self.kline_dir = Path(config["kline_dir"])
        self.output_dir = Path(config["output_dir"])

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
        passed = [r for r in all_results if r.get("total_score", 0) >= min_score]
        excluded = [r["code"] for r in all_results if r.get("total_score", 0) < min_score]

        passed.sort(key=lambda r: r.get("total_score", 0), reverse=True)

        recommendations = [
            {
                "rank": i + 1,
                "code": r["code"],
                "verdict": r.get("verdict", ""),
                "total_score": r.get("total_score", 0),
                "signal_type": r.get("signal_type", ""),
                "comment": r.get("comment", ""),
            }
            for i, r in enumerate(passed)
        ]

        return {
            "date": pick_date,
            "min_score_threshold": min_score,
            "total_reviewed": len(all_results),
            "recommendations": recommendations,
            "excluded": excluded,
        }

    def _process_single_stock(self, candidate: dict, pick_date: str, out_dir: Path) -> tuple[Optional[dict], Optional[str]]:
        """处理单只股票的分析，供多线程调用"""
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
            result = self.review_stock(
                code=code,
                day_chart=day_chart,
                prompt=self.prompt,
            )
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            # 在每个线程中加入请求间隔，而不是在主线程中
            request_delay = self.config.get("request_delay", 1)
            time.sleep(request_delay)  # 每个请求后休眠，控制API调用频率

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

        # 并发数配置，默认5并发，可在config中通过max_workers调整
        max_workers = self.config.get("max_workers", 5)
        request_delay = self.config.get("request_delay", 1)  # 单线程版本的delay是5，多线程版本降低到1，避免等待过久
        print(f"[INFO] 多线程模式已启用，并发数={max_workers}，请求间隔={request_delay}秒")

        # 先处理已经存在的缓存文件
        for candidate in candidates:
            code = candidate["code"]
            out_file = out_dir / f"{code}.json"
            if self.config.get("skip_existing", False) and out_file.exists():
                try:
                    with open(out_file, encoding="utf-8") as f:
                        result = json.load(f)
                    all_results.append(result)
                    print(f"[✅] {code} — 已缓存，跳过")
                except:
                    pass

        # 过滤出需要处理的股票
        to_process = [c for c in candidates if not (self.config.get("skip_existing", False) and (out_dir / f"{c['code']}.json").exists())]
        print(f"[INFO] 待分析股票数：{len(to_process)}")

        if to_process:
            # 多线程处理
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 绑定固定参数
                process_func = partial(self._process_single_stock, pick_date=pick_date, out_dir=out_dir)
                futures = [executor.submit(process_func, c) for c in to_process]

                # 处理结果
                completed = 0
                for future in as_completed(futures):
                    completed += 1
                    result, failed_code = future.result()
                    if result:
                        all_results.append(result)
                        verdict = result.get("verdict", "?")
                        score = result.get("total_score", "?")
                        print(f"[✅] [{completed}/{len(to_process)}] {result['code']} — 完成 | verdict={verdict}, score={score}")
                    if failed_code:
                        failed_codes.append(failed_code)

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
        print(f"[INFO] 汇总推荐已写入: {suggestion_file}")
        print(f"       推荐股票数（score≥{min_score}）: {len(suggestion['recommendations'])}")

        print("\n✅ 全部完成。")
        print(f"   输出目录: {out_dir}")
