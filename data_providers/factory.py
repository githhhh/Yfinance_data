from typing import Dict, Any, Type
from data_providers.base_provider import BaseDataProvider
from data_providers.yahoo_provider import YahooDataProvider
from data_providers.schwab_provider import SchwabDataProvider, SchwabCredentials


class DataProviderFactory:
    """数据提供者工厂类 (Provider Factory)
    
    根据 provider_type 返回对应的数据提供者适配器实例。
    默认返回 YahooDataProvider，确保原有调用逻辑零干扰。
    """

    _providers: Dict[str, Type[BaseDataProvider]] = {
        "yahoo": YahooDataProvider,
        "schwab": SchwabDataProvider,
    }

    @classmethod
    def register_provider(cls, name: str, provider_cls: Type[BaseDataProvider]) -> None:
        """注册新的数据提供者实现。"""
        cls._providers[name.lower()] = provider_cls

    @classmethod
    def get_provider(cls, provider_type: str = "yahoo", **kwargs: Any) -> BaseDataProvider:
        """根据名称构造并返回数据提供者实例。
        
        Args:
            provider_type: 数据源名称 ('yahoo' 或 'schwab')
            **kwargs: 传递给 Provider 构造函数的参数（如 credentials、batch_size 等）
        """
        key = provider_type.lower().strip()
        if key not in cls._providers:
            raise ValueError(
                f"未已知的数据源类型: '{provider_type}'。当前支持的数据源: {list(cls._providers.keys())}"
            )

        provider_cls = cls._providers[key]

        if key == "schwab":
            # 如果传了 app_key / token_path 等参数，自动构建 SchwabCredentials
            if "creds" not in kwargs and any(
                k in kwargs for k in ["app_key", "app_secret", "callback_url", "token_path"]
            ):
                kwargs["creds"] = SchwabCredentials(
                    app_key=kwargs.pop("app_key", None),
                    app_secret=kwargs.pop("app_secret", None),
                    callback_url=kwargs.pop("callback_url", None),
                    token_path=kwargs.pop("token_path", None),
                )
        else:
            # 清理 Schwab 特有的凭证 kwargs，防止传给其他 Provider (如 Yahoo) 导致 TypeError
            for cred_key in ["app_key", "app_secret", "callback_url", "token_path"]:
                kwargs.pop(cred_key, None)

        return provider_cls(**kwargs)

