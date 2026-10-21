"""Browser configuration never selects an implicit or remote debugging session."""
import pytest
from returns.result import Failure

from rai.actions.browser import CdpBrowserBackend, web_url
from rai.kernel.ports import CancellationToken


@pytest.mark.parametrize('endpoint', [
    'http://example.com:9222', 'http://127.0.0.1:9222/path',
    'http://user:password@127.0.0.1:9222', 'http://127.0.0.1:9222?secret=1',
    'http://127.0.0.1:9222#fragment', 'https://127.0.0.1:9222', 9222,
])
def test_invalid_debugging_endpoint_rejected(endpoint):
    with pytest.raises(ValueError):
        CdpBrowserBackend(endpoint)


async def test_browser_missing_configuration_is_explicit():
    backend = CdpBrowserBackend()
    token = CancellationToken()
    assert await backend.open('https://example.com', token) == Failure('BROWSER_NOT_CONFIGURED')
    assert await backend.read('https://example.com', token) == Failure('BROWSER_NOT_CONFIGURED')


@pytest.mark.parametrize('url', ['file:///etc/passwd', 'javascript:alert(1)',
                                 'https://user:password@example.com', 'data:text/html,hello'])
def test_result_urls_cannot_be_local_or_executable(url):
    assert not web_url(url)


async def test_browser_handles_are_scoped_and_local_context_never_searches(tmp_path):
    from returns.result import Success
    from rai.actions.browser import BrowserCapability, register_browser_capabilities
    from rai.actions.handles import SQLiteHandleStore
    from rai.kernel.capabilities import CapabilityRegistry
    from rai.kernel.records import DataClass, ProducerIdentity
    from rai.kernel.transport import normalize_request

    class Backend:
        searches = 0
        opened = []
        redirect = False

        async def search(self, query, token):
            self.searches += 1
            return Success(({'name': 'Example', 'url': 'https://example.com/'},))

        async def open(self, url, token):
            self.opened.append(url)
            return Success({'target_id': 'tab', 'observed_url': 'https://other.example/' if self.redirect else url})

    handles = SQLiteHandleStore(tmp_path / 'handles.db')
    backend = Backend()
    registry = CapabilityRegistry()
    register_browser_capabilities(registry, handles, backend)
    search = BrowserCapability('browser.search', handles, backend)
    opening = BrowserCapability('browser.open_result', handles, backend)
    request = normalize_request(registry.descriptor('browser.search'), {'task_id': 'public-task', 'query': 'example'})
    denied = await search.invoke(request, CancellationToken())
    assert denied.failure().code == 'PUBLIC_CONTEXT_REQUIRED'
    assert backend.searches == 0
    request = request.model_copy(update={'data_class': DataClass.PUBLIC})
    found = await search.invoke(request, CancellationToken())
    handle = found.unwrap().output['results'][0]['handle']
    request = normalize_request(registry.descriptor('browser.open_result'),
                                {'task_id': 'public-task', 'handle': handle}, data_class=DataClass.PUBLIC)
    intruder = ProducerIdentity(producer_id='another-user', kind='user', version='1.0.0')
    denied = await opening.invoke(request.model_copy(update={'actor': intruder}), CancellationToken())
    assert isinstance(denied, Failure)
    wrong_task = request.model_copy(update={'arguments': {'task_id': 'other-task', 'handle': handle}})
    assert isinstance(await opening.invoke(wrong_task, CancellationToken()), Failure)
    assert backend.opened == []
    verified = await opening.invoke(request, CancellationToken())
    assert verified.unwrap().verification['observed_url'] == 'https://example.com/'
    backend.redirect = True
    redirected = await opening.invoke(request, CancellationToken())
    assert redirected.failure().code == 'POSTCONDITION_FAILED'


@pytest.mark.parametrize('url', [
    'http://localhost',
    'http://localhost:8080/path',
    'http://foo.local',
    'http://internal.lan',
    'http://127.0.0.1',
    'http://127.0.0.1:8080',
    'http://192.168.1.1',
    'http://10.0.0.1',
    'http://172.16.0.1',
    'http://169.254.169.254',
    'http://0.0.0.0',
    'http://[::1]',
    'http://[::ffff:127.0.0.1]',
    'http://[fe80::1]',
])
def test_web_url_blocks_private_and_local_addresses(url):
    assert not web_url(url)


async def test_fetch_http_page_snapshot_rejects_private_ips():
    from rai.actions.browser import _fetch_http_page_snapshot
    token = CancellationToken()
    assert (await _fetch_http_page_snapshot("http://127.0.0.1:8080/", token)) == Failure("STALE_RESOURCE")
    assert (await _fetch_http_page_snapshot("http://localhost:3000/", token)) == Failure("STALE_RESOURCE")
    assert (await _fetch_http_page_snapshot("http://169.254.169.254/meta-data", token)) == Failure("STALE_RESOURCE")


async def test_fetch_http_page_snapshot_rejects_dns_resolving_to_private_ip(monkeypatch):
    from rai.actions.browser import _fetch_http_page_snapshot
    token = CancellationToken()

    async def mock_resolve_private(*_args, **_kwargs):
        return False

    monkeypatch.setattr("rai.actions.browser._resolve_and_validate_public_host", mock_resolve_private)
    res = await _fetch_http_page_snapshot("https://public-looking-domain.com/page", token)
    assert res == Failure("STALE_RESOURCE")


async def test_fetch_http_page_snapshot_rejects_redirect_to_private_target(monkeypatch):
    import httpx
    from rai.actions.browser import _fetch_http_page_snapshot
    token = CancellationToken()

    async def mock_resolve(hostname, port):
        return True

    monkeypatch.setattr("rai.actions.browser._resolve_and_validate_public_host", mock_resolve)

    def mock_transport_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"Location": "http://169.254.169.254/latest/meta-data"})

    transport = httpx.MockTransport(mock_transport_handler)
    real_client = httpx.AsyncClient
    monkeypatch.setattr("rai.actions.browser.httpx.AsyncClient", lambda **kwargs: real_client(transport=transport, **kwargs))

    res = await _fetch_http_page_snapshot("https://example.com/start", token)
    assert res == Failure("STALE_RESOURCE")


async def test_fetch_http_page_snapshot_bounds_redirect_chains(monkeypatch):
    import httpx
    from rai.actions.browser import _fetch_http_page_snapshot
    token = CancellationToken()

    async def mock_resolve(hostname, port):
        return True

    monkeypatch.setattr("rai.actions.browser._resolve_and_validate_public_host", mock_resolve)

    step = 0

    def mock_transport_handler(request: httpx.Request) -> httpx.Response:
        nonlocal step
        step += 1
        return httpx.Response(302, headers={"Location": f"https://example.com/step-{step}"})

    transport = httpx.MockTransport(mock_transport_handler)
    real_client = httpx.AsyncClient
    monkeypatch.setattr("rai.actions.browser.httpx.AsyncClient", lambda **kwargs: real_client(transport=transport, **kwargs))

    res = await _fetch_http_page_snapshot("https://example.com/start", token)
    assert res == Failure("BROWSER_READ_FAILED")

