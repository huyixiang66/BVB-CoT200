"""Explicit transport selection; subscription never reuses a paid API credential."""
from dataclasses import dataclass, field
import os
from urllib.parse import urlsplit

PROVIDERS=('openai','boyue','subscription','subscription_direct')


@dataclass(frozen=True)
class Connection:
    provider: str
    base_url: str
    key: str = field(repr=False)

    def token_kwargs(self, cap):
        if cap is not None and (type(cap) is not int or cap<1):
            raise ValueError('Output token cap must be a positive integer')
        if self.provider in ('subscription','subscription_direct') and cap is not None:
            raise ValueError('Subscription transport cannot enforce an output token cap; omit --max-tokens')
        return {} if cap is None else {'max_completion_tokens':cap}

    def metadata(self):
        # A fork URL can contain a capability/session ID: never log it.
        return {'provider':self.provider,'transport':'responses_stream' if self.provider=='subscription_direct' else ('local_codex_proxy' if self.provider=='subscription' else 'chat_completions'),
                'usage_basis':'subscription_reported_tokens' if self.provider in ('subscription','subscription_direct') else 'api_reported_tokens'}


def resolve(provider=None, env_file=None, key_env=None, environ=None):
    from dotenv import dotenv_values
    cfg={**(dotenv_values(env_file) if env_file else {}), **(dict(os.environ) if environ is None else environ)}
    provider=provider or cfg.get('BVB_API_PROVIDER') or 'openai'
    if provider not in PROVIDERS:
        raise ValueError('Unknown API provider')
    if provider=='subscription_direct':
        if key_env:raise ValueError('Direct subscription uses OAuth, not key-env')
        import importlib.util
        from pathlib import Path
        helper=Path(__file__).resolve().parents[1]/'scripts/subscription_direct_client.py'
        spec=importlib.util.spec_from_file_location('bvb_direct_oauth',helper)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return Connection(provider,'https://api.openai.com/v1',module.access_token())
    if provider=='subscription':
        if key_env and key_env!='BVB_PROXY_API_KEY':
            raise ValueError('Subscription must use its separate BVB_PROXY_API_KEY')
        key=cfg.get('BVB_PROXY_API_KEY')
        url=cfg.get('BVB_SUBSCRIPTION_BASE_URL') or 'http://127.0.0.1:18317/v1'
        parsed=urlsplit(url)
        if parsed.scheme!='http' or parsed.hostname not in ('127.0.0.1','localhost','::1') or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path.rstrip('/')!='/v1':
            raise ValueError('Subscription base must be a local http://loopback:port/v1 endpoint')
        if not key or len(key)<24:
            raise ValueError('Missing local BVB_PROXY_API_KEY (at least 24 characters); never use LLM_API_KEY')
    elif provider=='boyue':
        key=cfg.get(key_env or 'BOYUE_API_KEY')
        url=cfg.get('BOYUE_BASE_URL')
        if not url:raise ValueError('Missing BOYUE_BASE_URL')
    else:
        key=cfg.get(key_env or 'LLM_API_KEY')
        url='https://api.openai.com/v1'
    if not key:raise ValueError('Missing credential for '+provider)
    return Connection(provider,url.rstrip('/'),key)


def create_client(connection,timeout=600):
    if connection.provider=='subscription_direct':
        from .direct_subscription import DirectClient
        return DirectClient(connection,timeout)
    from openai import OpenAI
    return OpenAI(api_key=connection.key,base_url=connection.base_url,timeout=timeout,max_retries=0)
