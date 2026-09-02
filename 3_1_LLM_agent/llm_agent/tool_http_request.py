import ipaddress
import json
import socket
from urllib.parse import urlparse

import requests


class HTTPRequestTool:
    """Инструмент для выполнения безопасных GET/POST HTTP-запросов."""

    name = "http_request"
    description = (
        "Выполняет GET или POST HTTP-запрос. "
        "Формат: 'GET https://example.com' или "
        '\'POST https://example.com {"key": "value"}\'.'
    )

    ALLOWED_METHODS = {"GET", "POST"}
    TIMEOUT = 10

    def use(self, request_string: str) -> str:
        """
        Выполняет HTTP-запрос.

        Примеры:
            GET https://api.ipify.org
            GET https://httpbin.org/get {"name": "Alex"}
            POST https://httpbin.org/post {"name": "Alex"}
        """

        try:
            method, url, params = self._parse_request(request_string)
            self._validate_url(url)

            if method == "GET":
                response = requests.get(
                    url,
                    params=params,
                    timeout=self.TIMEOUT,
                    allow_redirects=False,
                )
            else:
                response = requests.post(
                    url,
                    json=params,
                    timeout=self.TIMEOUT,
                    allow_redirects=False,
                )

            response.raise_for_status()
            return response.text

        except ValueError as exc:
            return f"Ошибка: {exc}"

        except requests.exceptions.RequestException as exc:
            return f"Ошибка HTTP-запроса: {exc}"

    def _parse_request(self, request_string: str):
        """Разбирает строку запроса на HTTP-метод, URL и параметры."""

        parts = request_string.strip().split(maxsplit=2)

        if len(parts) < 2:
            raise ValueError(
                "Ожидается формат: METHOD URL [JSON_PARAMS]"
            )

        method = parts[0].upper()
        url = parts[1]

        if method not in self.ALLOWED_METHODS:
            raise ValueError(
                f"Метод {method} не поддерживается. "
                "Разрешены только GET и POST."
            )

        params = {}

        if len(parts) == 3:
            try:
                params = json.loads(parts[2])
            except json.JSONDecodeError as exc:
                raise ValueError("Параметры должны быть валидным JSON") from exc

            if not isinstance(params, dict):
                raise ValueError(
                    "Параметры запроса должны быть JSON-объектом"
                )

        return method, url, params

    def _validate_url(self, url: str):
        """
        Проверяет URL и запрещает запросы во внутреннюю сеть.

        Это защищает инструмент от простых SSRF-атак, когда LLM
        пытаются заставить обратиться к localhost или внутренним адресам.
        """

        parsed = urlparse(url)

        if parsed.scheme not in ("http", "https"):
            raise ValueError(
                "Разрешены только URL со схемой http или https"
            )

        if not parsed.hostname:
            raise ValueError("URL не содержит имя хоста")

        hostname = parsed.hostname

        if hostname.lower() == "localhost":
            raise ValueError("Запросы к localhost запрещены")

        try:
            addresses = socket.getaddrinfo(
                hostname,
                None,
                proto=socket.IPPROTO_TCP,
            )
        except socket.gaierror as exc:
            raise ValueError(
                f"Не удалось определить IP-адрес хоста {hostname}"
            ) from exc

        for address in addresses:
            ip = ipaddress.ip_address(address[4][0])

            if (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_multicast
                or ip.is_reserved
                or ip.is_unspecified
            ):
                raise ValueError(
                    f"Запросы к внутренним адресам запрещены: {ip}"
                )
