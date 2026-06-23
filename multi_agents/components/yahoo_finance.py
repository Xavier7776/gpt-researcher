"""
Yahoo Finance 数据获取模块
----------------------------
独立的金融数据获取工具，通过 yfinance 库获取美股/港股/A 股基本面数据。
不耦合进任何 Agent 类，作为纯工具模块供 Researcher 调用。

使用方式:
    tool = YahooFinanceTool("AAPL")
    overview = await tool.get_stock_overview()
    statements = await tool.get_financial_statements()
    peers = await tool.get_industry_peers()
"""

import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# --- 纯函数：从 query 文本中提取 ticker ---

# 常见美股 ticker → 公司名映射（A 股/港股暂不支持 yfinance）
_TICKER_PATTERNS: dict[str, str] = {
    "AAPL": "苹果",
    "GOOGL": "谷歌",
    "GOOG": "谷歌",
    "MSFT": "微软",
    "AMZN": "亚马逊",
    "META": "Meta",
    "NVDA": "英伟达",
    "TSLA": "特斯拉",
    "NFLX": "奈飞",
    "AMD": "AMD",
    "INTC": "英特尔",
    "BABA": "阿里巴巴",
    "JD": "京东",
    "PDD": "拼多多",
    "NIO": "蔚来",
    "BIDU": "百度",
    "TSM": "台积电",
    "DIS": "迪士尼",
    "BA": "波音",
    "JPM": "摩根大通",
    "GS": "高盛",
    "V": "Visa",
    "MA": "万事达",
    "WMT": "沃尔玛",
    "COST": "好市多",
    "KO": "可口可乐",
    "PEP": "百事",
    "JNJ": "强生",
    "PFE": "辉瑞",
}


def extract_ticker_from_query(query: str) -> Optional[str]:
    """从用户查询文本中提取股票代码。

    匹配策略（按优先级）：
    1. 括号内的全大写字母（如 "苹果公司(AAPL)"）
    2. 已知 ticker 直接匹配（如 "AAPL"）
    3. 已知中文公司名反向匹配

    Args:
        query: 用户输入的查询文本

    Returns:
        ticker 字符串（如 "AAPL"），未匹配到返回 None
    """
    import re

    # 策略 1：括号包裹的全大写 1-5 字母（如 (AAPL)）
    bracket_match = re.search(r'\(([A-Z]{1,5})\)', query)
    if bracket_match:
        ticker = bracket_match.group(1)
        if ticker in _TICKER_PATTERNS or len(ticker) <= 5:
            return ticker


    # 策略 2：纯大写 1-5 字母独立出现（如 "分析 AAPL 的..."）
    #\b 就是给正则引擎加的一个卡尺，用来卡住匹配内容的左右两端，确保被匹配的内容是一个孤立的、完整的单词。
    standalone_match = re.search(r'\b([A-Z]{1,5})\b', query)
    if standalone_match:
        ticker = standalone_match.group(1)
        if ticker in _TICKER_PATTERNS:
            return ticker

    # 策略 3：中文公司名反向匹配
    for ticker, name in _TICKER_PATTERNS.items():
        if name in query and ticker not in {"GOOG", "GOOGL"}:
            return ticker

    # 策略 4：英文公司名匹配
    name_lower = query.lower()
    _NAME_TO_TICKER = {
        "apple": "AAPL", "microsoft": "MSFT", "google": "GOOGL",
        "amazon": "AMZN", "meta": "META", "nvidia": "NVDA",
        "tesla": "TSLA", "netflix": "NFLX", "amd": "AMD",
        "intel": "INTC",
    }
    for name, ticker in _NAME_TO_TICKER.items():
        if name in name_lower:
            return ticker

    return None


# --- 数据类：Yahoo Finance 工具封装 ---

class YahooFinanceTool:
    """Yahoo Finance 数据获取工具。

    封装 yfinance 的同步调用，通过 asyncio.to_thread 提供异步接口。
    内部缓存已获取的数据，避免同一 ticker 重复请求。

    Attributes:
        ticker: 股票代码（如 "AAPL"）
        _cache: 内部缓存 dict
    """

    def __init__(self, ticker: str):
        #股票代码
        self.ticker = ticker.upper().strip()
        self._cache: dict = {}

    @property
    def yf_ticker(self):
        """懒加载 yfinance Ticker 对象（缓存避免重复创建）。"""
        if "_yf_ticker" not in self._cache:
            import yfinance as yf
            #返回股票对象
            self._cache["_yf_ticker"] = yf.Ticker(self.ticker)
        return self._cache["_yf_ticker"]

    # ------------------------------------------------------------------
    # 1. 股票基本面概览
    # ------------------------------------------------------------------

    async def get_stock_overview(self) -> dict:
        """获取股票基本面和估值概况。

        Returns:
            {
                ticker, name, price, market_cap, pe_ratio, pb_ratio,
                roe, revenue_growth, debt_to_equity, profit_margin,
                dividend_yield, beta, sector, industry, summary,
                currency, exchange, fifty_two_week_high, fifty_two_week_low
            }
            获取失败返回空 dict
        """
        cache_key = "stock_overview"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            info = await asyncio.to_thread(self._fetch_info)

            result = {
                "ticker": self.ticker,
                "name": info.get("longName") or info.get("shortName", ""),
                "price": info.get("currentPrice") or info.get("regularMarketPrice"),
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE") or info.get("forwardPE"),
                "pb_ratio": info.get("priceToBook"),
                "roe": self._pct(info.get("returnOnEquity")),
                "revenue_growth": self._pct(info.get("revenueGrowth")),
                "debt_to_equity": self._pct(info.get("debtToEquity")),
                "profit_margin": self._pct(info.get("profitMargins")),
                "dividend_yield": self._pct(info.get("dividendYield")),
                "beta": info.get("beta"),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "summary": info.get("longBusinessSummary", ""),
                "currency": info.get("currency", "USD"),
                "exchange": info.get("exchange", ""),
                "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
                "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
            }

            self._cache[cache_key] = result
            return result

        except Exception as e:
            logger.warning(f"[YahooFinance] get_stock_overview({self.ticker}) 失败: {e}")
            return {}

    # ------------------------------------------------------------------
    # 2. 财务报表数据
    # ------------------------------------------------------------------

    async def get_financial_statements(self) -> dict:
        """获取财报关键数据（近 4 个季度 + 近 4 年）。

        Returns:
            {
                quarterly_revenue: [{date, revenue} * 4],
                quarterly_earnings: [{date, earnings} * 4],
                annual_revenue: [{date, revenue} * 4],
                annual_earnings: [{date, earnings} * 4],
                cash_flow: {operating_cash_flow, free_cash_flow, ...},
                balance_sheet: {total_assets, total_debt, ...}
            }
            获取失败返回空 dict
        """
        cache_key = "financial_statements"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            result = await asyncio.to_thread(self._fetch_financials)
            self._cache[cache_key] = result
            return result

        except Exception as e:
            logger.warning(f"[YahooFinance] get_financial_statements({self.ticker}) 失败: {e}")
            return {}

    def _fetch_financials(self) -> dict:
        """同步获取财报数据（在 executor 中运行）。"""
        #获得股票对象
        t = self.yf_ticker

        # 季度数据
        try:
            #月度财务数据
            q_fin = t.quarterly_financials
            q_rev = []
            q_earn = []
            #获得前4个季度
            for col in list(q_fin.columns)[:4]:
                date_str = col.strftime("%Y-%m-%d") if hasattr(col, "strftime") else str(col)
                #df.loc[行标签, 列标签],取哪一行哪一列
                rev = q_fin.loc["Total Revenue", col] if "Total Revenue" in q_fin.index else None
                earn = q_fin.loc["Net Income", col] if "Net Income" in q_fin.index else None
                q_rev.append({"date": date_str, "revenue": _safe_float(rev)})
                q_earn.append({"date": date_str, "earnings": _safe_float(earn)})
        except Exception:
            q_rev, q_earn = [], []

        # 年度数据
        try:
            a_fin = t.financials
            a_rev = []
            a_earn = []
            for col in list(a_fin.columns)[:4]:
                date_str = col.strftime("%Y-%m-%d") if hasattr(col, "strftime") else str(col)
                rev = a_fin.loc["Total Revenue", col] if "Total Revenue" in a_fin.index else None
                earn = a_fin.loc["Net Income", col] if "Net Income" in a_fin.index else None
                a_rev.append({"date": date_str, "revenue": _safe_float(rev)})
                a_earn.append({"date": date_str, "earnings": _safe_float(earn)})
        except Exception:
            a_rev, a_earn = [], []

        # 现金流
        try:
            cf = t.cashflow
            cf_col = cf.columns[0] if len(cf.columns) > 0 else None
            cash_flow = {
                "operating_cash_flow": _safe_float(cf.loc["Operating Cash Flow", cf_col]) if cf_col is not None and "Operating Cash Flow" in cf.index else None,
                "free_cash_flow": _safe_float(cf.loc["Free Cash Flow", cf_col]) if cf_col is not None and "Free Cash Flow" in cf.index else None,
                "capital_expenditure": _safe_float(cf.loc["Capital Expenditure", cf_col]) if cf_col is not None and "Capital Expenditure" in cf.index else None,
            }
        except Exception:
            cash_flow = {}

        # 资产负债表
        try:
            bs = t.balance_sheet
            bs_col = bs.columns[0] if len(bs.columns) > 0 else None
            balance_sheet = {
                "total_assets": _safe_float(bs.loc["Total Assets", bs_col]) if bs_col is not None and "Total Assets" in bs.index else None,
                "total_debt": _safe_float(bs.loc["Total Debt", bs_col]) if bs_col is not None and "Total Debt" in bs.index else None,
                "total_equity": _safe_float(bs.loc["Stockholders Equity", bs_col]) if bs_col is not None and "Stockholders Equity" in bs.index else None,
                "current_assets": _safe_float(bs.loc["Current Assets", bs_col]) if bs_col is not None and "Current Assets" in bs.index else None,
                "current_liabilities": _safe_float(bs.loc["Current Liabilities", bs_col]) if bs_col is not None and "Current Liabilities" in bs.index else None,
            }
        except Exception:
            balance_sheet = {}

        return {
            "quarterly_revenue": q_rev,
            "quarterly_earnings": q_earn,
            "annual_revenue": a_rev,
            "annual_earnings": a_earn,
            "cash_flow": cash_flow,
            "balance_sheet": balance_sheet,
        }

    # ------------------------------------------------------------------
    # 3. 同行业可比公司
    # ------------------------------------------------------------------

    async def get_industry_peers(self) -> list[dict]:
        """获取同行业可比公司列表及关键指标。

        先通过 yfinance 获取推荐的可比公司列表，
        再并行拉取每家的关键估值指标。

        Returns:
            [{ticker, name, market_cap, pe, pb, roe, revenue_growth}, ...]
            获取失败返回空列表
        """
        cache_key = "industry_peers"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            # 获取可比公司 ticker 列表
            symbol_list = await asyncio.to_thread(self._fetch_peer_symbols)

            if not symbol_list:
                self._cache[cache_key] = []
                return []

            # 并行获取每家公司的关键指标（限制前 6 家）
            limited = symbol_list[:6]
            tasks = [self._fetch_peer_info(s) for s in limited]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            peers = [r for r in results if isinstance(r, dict) and r]

            self._cache[cache_key] = peers
            return peers

        except Exception as e:
            logger.warning(f"[YahooFinance] get_industry_peers({self.ticker}) 失败: {e}")
            return []

    def _fetch_peer_symbols(self) -> list[str]:
        """同步获取可比公司 ticker 列表。

        优先使用 yfinance Search 按行业搜索（过滤掉 ETF/指数），
        失败时回退到预设的行业可比公司映射。
        """
        import yfinance as yf

        # 方法 1：使用 yfinance Search 按行业搜索
        try:
            info = self.yf_ticker.info
            industry = info.get("industry", "")
            sector = info.get("sector", "")
            query = industry or sector

            if query:
                search = yf.Search(query)
                quotes = search.quotes if hasattr(search, "quotes") else []
                symbols = []
                for q in quotes[:20]:
                    sym = q.get("symbol", "")
                    qtype = q.get("quoteType", "").upper()
                    # 只保留普通股票，过滤 ETF/指数/基金
                    if (sym and sym != self.ticker
                            and "." not in sym
                            and "^" not in sym
                            and "-" not in sym
                            and qtype in ("EQUITY", "")):
                        symbols.append(sym)
                if symbols:
                    return symbols[:10]
        except Exception:
            pass

        # 方法 2：利用 recommendations 中的相关股票
        try:
            recs = self.yf_ticker.recommendations
            if recs is not None and not recs.empty:
                symbols = []
                seen = set()
                for _, row in recs.iterrows():
                    # recommendations 通常包含目标公司，取同行业分析师的覆盖范围
                    pass  # 结构不稳定，跳过
        except Exception:
            pass

        # 备用：根据已知行业映射返回预设可比公司
        return _get_fallback_peers(self.ticker)

    async def _fetch_peer_info(self, symbol: str) -> dict:
        """异步获取单家公司的关键指标。"""
        try:
            info = await asyncio.to_thread(
                lambda s=symbol: _fetch_single_info(s)
            )
            return info
        except Exception:
            return {}

    def _fetch_info(self) -> dict:
        """同步获取 ticker info。"""
        return self.yf_ticker.info

    @staticmethod
    def _pct(value) -> Optional[float]:
        """将 yfinance 返回的比例值转为百分比（保留 2 位）。"""
        if value is None:
            return None
        try:
            return round(float(value) * 100, 2)
        except (TypeError, ValueError):
            return None


# --- 辅助函数 ---

def _safe_float(value) -> Optional[float]:
    """安全转换数值，None 或不合法返回 None。"""
    if value is None:
        return None
    try:
        v = float(value)
        return v if v == v else None  # NaN check
    except (TypeError, ValueError):
        return None


def _fetch_single_info(symbol: str) -> dict:
    """同步获取单个公司 info 并提取关键字段。"""
    import yfinance as yf

    t = yf.Ticker(symbol)
    info = t.info or {}

    return {
        "ticker": symbol,
        "name": info.get("shortName") or info.get("longName", symbol),
        "market_cap": info.get("marketCap"),
        "pe": info.get("trailingPE") or info.get("forwardPE"),
        "pb": info.get("priceToBook"),
        "roe": _safe_pct(info.get("returnOnEquity")),
        "revenue_growth": _safe_pct(info.get("revenueGrowth")),
    }


def _safe_pct(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return round(float(value) * 100, 2)
    except (TypeError, ValueError):
        return None


# 备用：当 yfinance 搜索不可用时，返回预设的行业可比公司
_FALLBACK_PEERS: dict[str, list[str]] = {
    "AAPL": ["MSFT", "GOOGL", "DELL", "HPQ", "SONO"],
    "MSFT": ["AAPL", "GOOGL", "AMZN", "ORCL", "CRM"],
    "GOOGL": ["META", "MSFT", "AAPL", "SNAP", "PINS"],
    "AMZN": ["WMT", "TGT", "COST", "JD", "BABA"],
    "META": ["GOOGL", "SNAP", "PINS", "TWTR", "BIDU"],
    "NVDA": ["AMD", "INTC", "TSM", "QCOM", "AVGO"],
    "TSLA": ["F", "GM", "TM", "NIO", "RIVN"],
    "NFLX": ["DIS", "WBD", "CMCSA", "PARA", "ROKU"],
    "JPM": ["BAC", "WFC", "C", "GS", "MS"],
    "JNJ": ["PFE", "MRK", "ABBV", "BMY", "LLY"],
    "BABA": ["JD", "PDD", "AMZN", "VIPS", "BIDU"],
    "TSM": ["INTC", "AMD", "NVDA", "QCOM", "ASML"],
}


def _get_fallback_peers(ticker: str) -> list[str]:
    """获取预设的备用可比公司列表。"""
    return _FALLBACK_PEERS.get(ticker.upper(), [])
