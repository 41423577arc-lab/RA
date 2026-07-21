from contextlib import nullcontext

from mcp_server.project_repository import ProjectRepository


def make_repository_spy():
    repository = object.__new__(ProjectRepository)
    repository.session_factory = lambda: nullcontext(object())
    repository.threshold = 0.18
    calls: list[tuple] = []
    repository._add_exact_matches = (
        lambda session, matches, people, organizations: calls.append(
            ("exact", people, organizations)
        )
    )
    repository._add_text_matches = (
        lambda session, matches, terms: calls.append(("text", terms))
    )
    repository._add_vector_matches = (
        lambda session, matches, keywords: calls.append(("vector", keywords))
    )
    return repository, calls


def test_person_scope_does_not_expand_to_organization_or_keywords() -> None:
    repository, calls = make_repository_spy()

    results = repository.search(
        ["王传福"], ["比亚迪股份有限公司"], ["新能源", "储能"]
    )

    assert results == []
    assert calls == [
        ("exact", ["王传福"], []),
        ("text", ["王传福"]),
    ]


def test_organization_and_keyword_scope_is_used_without_a_person() -> None:
    repository, calls = make_repository_spy()

    results = repository.search([], ["比亚迪股份有限公司"], ["新能源"])

    assert results == []
    assert calls == [
        ("exact", [], ["比亚迪股份有限公司"]),
        ("text", ["比亚迪股份有限公司", "新能源"]),
        ("vector", ["新能源"]),
    ]
