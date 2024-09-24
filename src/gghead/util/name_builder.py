from typing import Optional

from elias.config import Config


class NameBuilder:
    def __init__(self, config: Config, default_config: Config, prefix: Optional[str] = None, suffix: Optional[str] = None):
        self._config = config
        self._default_config = default_config

        if prefix is None:
            self._name = ""
        else:
            self._name = prefix
        self._suffix = suffix

    def add(self, key: str, short: Optional[str] = None, always: bool= False):
        if short is None:
            short = key
        short = short.replace("_", '-')

        value = getattr(self._config, key)
        if value != getattr(self._default_config, key) or always:
            if isinstance(value, bool):
                if value:
                    name = short
                else:
                    name = f"no-{short}"
            elif isinstance(value, list) or isinstance(value, tuple) or isinstance(value, set):
                value = ','.join([str(v) for v in value])
                name = f"{short}-{value}"
            else:
                name = f"{short}-{value}"

            if self._name == '':
                self._name = name
            else:
                self._name += f"_{name}"

    def get(self) -> str:
        name = self._name
        if self._suffix is not None:
            name += self._suffix

        return name