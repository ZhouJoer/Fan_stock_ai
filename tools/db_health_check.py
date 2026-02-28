"""
检查 db/stock_data.db 数据重复与体积，并生成可视化 HTML 报告。
用法: python -m tools.db_health_check [--fix] [--output report.html]
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path

# 项目根目录
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DB_PATH = ROOT / "db" / "stock_data.db"


def get_table_info(conn: sqlite3.Connection) -> list[dict]:
    """获取所有表名（排除 sqlite_ 系统表）。"""
    cur = conn.cursor()
    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    )
    return [{"name": row[0]} for row in cur.fetchall()]


def get_page_count(conn: sqlite3.Connection) -> int:
    """数据库文件占用的页数。"""
    cur = conn.cursor()
    cur.execute("PRAGMA page_count")
    return cur.fetchone()[0]


def get_page_size(conn: sqlite3.Connection) -> int:
    """页大小（字节）。"""
    cur = conn.cursor()
    cur.execute("PRAGMA page_size")
    return cur.fetchone()[0]


def get_table_row_count(conn: sqlite3.Connection, table: str) -> int:
    cur = conn.cursor()
    cur.execute(f"SELECT COUNT(*) FROM [{table}]")
    return cur.fetchone()[0]


def get_table_duplicate_check(conn: sqlite3.Connection, table: str) -> tuple[int, int, int]:
    """
    返回 (总行数, 按主键去重后的行数, 重复行数)。
    根据已知表结构推断主键。
    """
    cur = conn.cursor()
    cur.execute(f"SELECT COUNT(*) FROM [{table}]")
    total = cur.fetchone()[0]
    if total == 0:
        return 0, 0, 0

    # 已知表的主键
    key_columns = {
        "stock_daily": ["symbol", "trade_date"],
        "stock_weekly": ["symbol", "trade_date"],
        "stock_monthly": ["symbol", "trade_date"],
        "kline_sync_state": ["symbol"],
        "factor_values": ["symbol", "trade_date", "factor_name"],
        "forward_returns": ["symbol", "trade_date", "label_horizon"],
    }
    cols = key_columns.get(table)
    if not cols:
        # 尝试用表中所有列做去重
        cur.execute(f"PRAGMA table_info([{table}])")
        cols = [row[1] for row in cur.fetchall()]
    if not cols:
        return total, total, 0

    # SQLite 的 COUNT(DISTINCT) 只接受单表达式，多列用子查询
    cols_str = ", ".join(f"[{c}]" for c in cols)
    cur.execute(f"SELECT COUNT(*) FROM (SELECT DISTINCT {cols_str} FROM [{table}])")
    distinct = cur.fetchone()[0]
    duplicates = max(0, total - distinct)
    return total, distinct, duplicates


def get_table_size_estimate(conn: sqlite3.Connection, table: str) -> int:
    """估算单表占用（通过 sqlite_stat1 或 简单估算）。"""
    cur = conn.cursor()
    try:
        cur.execute(f"SELECT SUM(pgsize) FROM dbstat WHERE name=?", (table,))
        row = cur.fetchone()
        if row and row[0] is not None:
            return row[0]
    except sqlite3.OperationalError:
        pass
    # 未开启 dbstat 时用行数 * 估计每行字节
    n = get_table_row_count(conn, table)
    return n * 80  # 粗略估计每行约 80 字节


def run_analysis(db_path: Path) -> dict:
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA quick_check")  # 快速完整性检查
    try:
        page_count = get_page_count(conn)
        page_size = get_page_size(conn)
        db_size_bytes = page_count * page_size
    finally:
        conn.close()

    conn = sqlite3.connect(str(db_path))
    tables = get_table_info(conn)
    results = []
    for t in tables:
        name = t["name"]
        total, distinct, dup = get_table_duplicate_check(conn, name)
        est_size = get_table_size_estimate(conn, name)
        results.append({
            "table": name,
            "total_rows": total,
            "distinct_rows": distinct,
            "duplicate_rows": dup,
            "estimated_size_bytes": est_size,
        })
    conn.close()

    return {
        "db_path": str(db_path),
        "db_size_bytes": page_count * page_size,
        "page_count": page_count,
        "page_size": page_size,
        "tables": results,
    }


def fix_duplicates(db_path: Path) -> dict:
    """删除重复行（保留每个主键的第一条）。仅处理 stock_daily/weekly/monthly。"""
    conn = sqlite3.connect(str(db_path))
    fixed = {}
    for table, key_cols in [
        ("stock_daily", ["symbol", "trade_date"]),
        ("stock_weekly", ["symbol", "trade_date"]),
        ("stock_monthly", ["symbol", "trade_date"]),
    ]:
        cur = conn.cursor()
        cur.execute(f"SELECT COUNT(*) FROM [{table}]")
        before = cur.fetchone()[0]
        # SQLite 没有 ROW_NUMBER 的 DELETE 子查询在旧版本中写法不同，用临时表去重
        cols = "symbol, trade_date, open, close, high, low, volume, pct_chg"
        cur.execute(f"""
            DELETE FROM [{table}]
            WHERE rowid NOT IN (
                SELECT MIN(rowid) FROM [{table}]
                GROUP BY {", ".join(key_cols)}
            )
        """)
        cur.execute(f"SELECT COUNT(*) FROM [{table}]")
        after = cur.fetchone()[0]
        fixed[table] = before - after
    conn.commit()
    conn.close()
    return fixed


def build_html_report(data: dict, output_path: Path) -> None:
    """生成带简单图表的 HTML 报告。"""
    size_mb = data["db_size_bytes"] / (1024 * 1024)
    tables = data["tables"]
    total_rows = sum(t["total_rows"] for t in tables)
    total_dup = sum(t["duplicate_rows"] for t in tables)

    # 表格行
    rows_html = ""
    for t in tables:
        dup_pct = (100.0 * t["duplicate_rows"] / t["total_rows"]) if t["total_rows"] else 0
        size_kb = t["estimated_size_bytes"] / 1024
        rows_html += f"""
        <tr>
            <td>{t["table"]}</td>
            <td>{t["total_rows"]:,}</td>
            <td>{t["distinct_rows"]:,}</td>
            <td>{t["duplicate_rows"]:,}</td>
            <td>{dup_pct:.1f}%</td>
            <td>{size_kb:.1f} KB</td>
        </tr>"""

    # 用 inline 的 Chart.js 画图：表行数柱状图、重复行数、各表行数占比（饼图）
    chart_labels = [t["table"] for t in tables]
    chart_rows = [t["total_rows"] for t in tables]
    chart_dups = [t["duplicate_rows"] for t in tables]
    dup_summary = "存在重复行，建议执行 --fix 去重后执行 VACUUM。" if total_dup else "未发现重复行，体积主要来自数据量（如 factor_values 表）。"

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>stock_data.db 健康检查报告</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        body {{ font-family: "Segoe UI", system-ui, sans-serif; margin: 24px; background: #1a1a2e; color: #eee; }}
        h1 {{ color: #e94560; }}
        .summary {{ display: flex; gap: 24px; flex-wrap: wrap; margin: 20px 0; }}
        .card {{ background: #16213e; padding: 16px 24px; border-radius: 8px; min-width: 160px; }}
        .card .value {{ font-size: 1.5rem; color: #0f3460; font-weight: bold; }}
        .card.warn .value {{ color: #e94560; }}
        table {{ border-collapse: collapse; width: 100%; max-width: 720px; margin: 16px 0; background: #16213e; border-radius: 8px; overflow: hidden; }}
        th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid #0f3460; }}
        th {{ background: #0f3460; color: #e94560; }}
        .charts {{ display: flex; flex-wrap: wrap; gap: 24px; margin: 24px 0; }}
        .chart-wrap {{ width: 400px; height: 280px; }}
        footer {{ margin-top: 32px; color: #666; font-size: 0.9rem; }}
    </style>
</head>
<body>
    <h1>📊 stock_data.db 健康检查报告</h1>
    <p><strong>数据库路径：</strong> <code>{data["db_path"]}</code></p>
    <div class="summary">
        <div class="card">
            <div class="label">数据库大小</div>
            <div class="value">{size_mb:.2f} MB</div>
        </div>
        <div class="card">
            <div class="label">总行数</div>
            <div class="value">{total_rows:,}</div>
        </div>
        <div class="card {'warn' if total_dup else ''}">
            <div class="label">重复行数</div>
            <div class="value">{total_dup:,}</div>
        </div>
        <div class="card">
            <div class="label">数据表数量</div>
            <div class="value">{len(tables)}</div>
        </div>
    </div>
    <p><strong>重复检查结论：</strong> {dup_summary}</p>

    <h2>各表统计</h2>
    <table>
        <thead>
            <tr>
                <th>表名</th>
                <th>总行数</th>
                <th>去重行数</th>
                <th>重复行数</th>
                <th>重复占比</th>
                <th>估算大小</th>
            </tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>

    <h2>可视化</h2>
    <div class="charts">
        <div class="chart-wrap">
            <canvas id="chartRows"></canvas>
        </div>
        <div class="chart-wrap">
            <canvas id="chartDups"></canvas>
        </div>
        <div class="chart-wrap">
            <canvas id="chartPie"></canvas>
        </div>
    </div>

    <footer>报告由 tools/db_health_check.py 生成。若存在重复，可运行 <code>python -m tools.db_health_check --fix</code> 去重后使用 VACUUM 收缩文件。</footer>

    <script>
        const labels = {chart_labels};
        new Chart(document.getElementById("chartRows"), {{
            type: "bar",
            data: {{
                labels,
                datasets: [{{ label: "总行数", data: {chart_rows}, backgroundColor: "#0f3460" }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{ title: {{ display: true, text: "各表行数" }} }},
                scales: {{ y: {{ beginAtZero: true }} }}
            }}
        }});
        new Chart(document.getElementById("chartDups"), {{
            type: "bar",
            data: {{
                labels,
                datasets: [{{ label: "重复行数", data: {chart_dups}, backgroundColor: "#e94560" }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{ title: {{ display: true, text: "各表重复行数" }} }},
                scales: {{ y: {{ beginAtZero: true }} }}
            }}
        }});
        const pieColors = ["#e94560", "#0f3460", "#533483", "#16c79a", "#f4a261", "#2a9d8f"];
        new Chart(document.getElementById("chartPie"), {{
            type: "doughnut",
            data: {{
                labels,
                datasets: [{{ data: {chart_rows}, backgroundColor: pieColors.slice(0, labels.length) }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{ title: {{ display: true, text: "各表行数占比" }} }}
            }}
        }});
    </script>
</body>
</html>"""
    output_path.write_text(html, encoding="utf-8")
    print(f"已生成报告: {output_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="检查 stock_data.db 重复与体积并生成可视化报告")
    parser.add_argument("--fix", action="store_true", help="删除 stock_daily/weekly/monthly 中的重复行")
    parser.add_argument("--output", "-o", default="db_health_report.html", help="输出 HTML 报告路径")
    parser.add_argument("--db", default=None, help="数据库路径，默认 db/stock_data.db")
    args = parser.parse_args()

    db_path = Path(args.db) if args.db else DB_PATH
    if not db_path.is_file():
        print(f"数据库文件不存在: {db_path}")
        sys.exit(1)

    if args.fix:
        print("正在删除重复行...")
        fixed = fix_duplicates(db_path)
        for table, removed in fixed.items():
            print(f"  {table}: 删除 {removed} 行重复")
        print("建议随后执行 VACUUM 以收缩文件: sqlite3 db/stock_data.db 'VACUUM;'")

    data = run_analysis(db_path)
    size_mb = data["db_size_bytes"] / (1024 * 1024)
    print(f"\n数据库: {db_path}")
    print(f"大小: {size_mb:.2f} MB (页数 {data['page_count']}, 页大小 {data['page_size']} B)")
    print("\n表名              总行数    去重行数  重复行数  重复占比")
    print("-" * 60)
    for t in data["tables"]:
        dup_pct = (100.0 * t["duplicate_rows"] / t["total_rows"]) if t["total_rows"] else 0
        print(f"{t['table']:<18} {t['total_rows']:>10,} {t['distinct_rows']:>10,} {t['duplicate_rows']:>10,} {dup_pct:>6.1f}%")

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    build_html_report(data, out_path)


if __name__ == "__main__":
    main()
