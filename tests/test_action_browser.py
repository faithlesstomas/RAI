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
