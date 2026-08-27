from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, List, Optional, Tuple


class BaseDataProvider(ABC):
    """数据提供者抽象基类 (Data Provider Abstract Base Class)
    
    所有数据源适配器（Yahoo, Schwab 等）均必须继承此类，
    并实现 K 线行情下载与实时行情/期权查询抽象接口。
    """

    @abstractmethod
    def download_single_stock(
        self, symbol: str, period: str = "1y", interval: str = "1d"
    ) -> Tuple[str, Optional[pd.DataFrame]]:
        """抓取单只标的的 K 线行情数据。
        
        返回 (symbol, DataFrame)，DataFrame 必须遵循包含列:
        ['Open', 'High', 'Low', 'Close', 'Volume']，价格精度保持源数据原样。
        若抓取失败或无数据则 DataFrame 为 None。
        """
        pass

    @abstractmethod
    def download_batch_stocks(
        self, symbols: List[str], period: str = "1y", interval: str = "1d"
    ) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
        """批量并发抓取标的 K 线行情数据。
        
        返回 (all_data, failed_symbols)，其中 all_data 为 {symbol: DataFrame}。
        """
        pass

    @abstractmethod
    def fetch_option_chain(self, symbol: str) -> Optional[Dict]:
        """获取指定标的的期权链数据 (仅供接口调研扩展，绝对不写入 results_pkl)。"""
        pass

    @abstractmethod
    def fetch_quote(self, symbol: str) -> Optional[Dict]:
        """获取指定标的的交易日实时/最新点查行情快照 (绝对不写入 results_pkl)。"""
        pass
