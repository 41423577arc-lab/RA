from pathlib import Path

from app.schemas.task import ExtractedInfo, Person, WebPage
from app.services.extractor import RuleExtractor


SEED_DIR = Path(__file__).resolve().parents[2] / "seed"


def test_extracts_fixed_demo_entities() -> None:
    extractor = RuleExtractor(SEED_DIR)
    result = extractor.extract(
        "老板周五要和比亚迪股份有限公司的王传福董事长兼总裁吃饭，主要聊新能源和储能项目。"
    )

    assert result.event_type == "宴请"
    assert result.event_time == "周五"
    assert result.people == [
        Person(name="王传福", organization="比亚迪股份有限公司", title="董事长兼总裁")
    ]
    assert result.keywords == ["新能源", "储能"]


def test_extracts_multiple_people_without_duplicates() -> None:
    extractor = RuleExtractor(SEED_DIR)
    result = extractor.extract(
        "王传福董事长兼总裁和比亚迪股份有限公司参加会议，华星能源集团的李明副总经理也参加。"
    )

    names = [person.name for person in result.people]
    assert names == ["王传福", "李明"]
    assert len(names) == len(set(names))


def test_extracts_title_before_known_person_name() -> None:
    extractor = RuleExtractor(SEED_DIR)
    result = extractor.extract(
        "公司老板参加宴请，关键人物是比亚迪股份有限公司董事长王传福。"
    )

    assert result.people == [
        Person(name="王传福", organization="比亚迪股份有限公司", title="董事长")
    ]


def test_public_claims_are_original_sentences_with_sources() -> None:
    extractor = RuleExtractor(SEED_DIR)
    extracted = ExtractedInfo(
        event_type="宴请",
        people=[Person(name="王传福", organization="比亚迪股份有限公司", title="董事长")],
        keywords=["新能源", "储能"],
    )
    sentence = "王传福表示比亚迪股份有限公司持续发展新能源汽车、储能和电子等业务"
    pages = [
        WebPage(
            title="公司简介",
            url="https://example.com/byd",
            raw_content=(
                f"<p>{sentence}。</p>"
                "<p>比亚迪股份有限公司其他高管负责新能源汽车业务。</p>"
            ),
            rank=0,
        )
    ]

    claims = extractor.extract_public_claims(pages, extracted)

    assert claims
    assert claims[0].claim == sentence
    assert claims[0].source_url == "https://example.com/byd"
    assert "新能源" in claims[0].matched_keywords
    assert all("王传福" in claim.claim for claim in claims)


def test_public_claims_fall_back_to_organization_when_person_is_unknown() -> None:
    extractor = RuleExtractor(SEED_DIR)
    extracted = ExtractedInfo(
        event_type="会议",
        people=[Person(organization="华星能源集团")],
        keywords=["光伏"],
    )
    pages = [
        WebPage(
            title="企业动态",
            url="https://example.com/energy",
            raw_content="华星能源集团正在建设光伏发电项目。",
            rank=0,
        )
    ]

    claims = extractor.extract_public_claims(pages, extracted)

    assert len(claims) == 1
    assert claims[0].subject == "华星能源集团"
