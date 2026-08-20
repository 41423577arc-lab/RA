import csv
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import jieba
from bs4 import BeautifulSoup

from app.schemas.task import ExtractedInfo, Person, PublicClaim, WebPage


SENTENCE_SPLIT = re.compile(r"[。！？!?；;\n]+")
TIME_PATTERN = re.compile(
    r"今天|明天|后天|周[一二三四五六日天]|星期[一二三四五六日天]|"
    r"\d{4}年\d{1,2}月\d{1,2}日|\d{1,2}月\d{1,2}日"
)
PERSON_PATTERN = re.compile(
    r"([\u4e00-\u9fa5]{2,4})(?=董事长|总经理|副总经理|总裁|副总裁|局长|主任|书记|院长)"
)
ORG_PATTERN = re.compile(
    r"([\u4e00-\u9fa5]{2,30}?(?:集团|公司|银行|局|委员会|中心|研究院|大学))"
)
LOCATION_PATTERN = re.compile(r"(?:在|地点[:：])([^，。；;]{2,30})")


@dataclass(frozen=True)
class EntityEntry:
    candidate_id: str
    person_name: str
    person_alias: str
    organization_name: str
    organization_alias: str
    title: str
    region: str


class RuleExtractor:
    def __init__(self, seed_dir: Path):
        self.entities = self._load_entities(seed_dir / "entities.csv")
        self.titles = self._load_lines(seed_dir / "titles.txt")
        self.business_keywords = self._load_lines(seed_dir / "business_keywords.txt")
        for keyword in self.business_keywords:
            jieba.add_word(keyword, freq=2_000_000)

    @staticmethod
    def _load_lines(path: Path) -> list[str]:
        with path.open("r", encoding="utf-8") as handle:
            return sorted({line.strip() for line in handle if line.strip()}, key=len, reverse=True)

    @staticmethod
    def _load_entities(path: Path) -> list[EntityEntry]:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [EntityEntry(**row) for row in csv.DictReader(handle)]

    @staticmethod
    def normalize(text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        return re.sub(r"\s+", " ", text).strip()

    def extract(self, text: str) -> ExtractedInfo:
        normalized = self.normalize(text)
        event_type = "其他"
        for target, words in (
            ("宴请", ("吃饭", "饭局", "宴请", "赴宴")),
            ("拜访", ("拜访", "走访", "到访")),
            ("会议", ("开会", "会议", "座谈", "论坛")),
        ):
            if any(word in normalized for word in words):
                event_type = target
                break

        time_match = TIME_PATTERN.search(normalized)
        location_match = LOCATION_PATTERN.search(normalized)
        people = self._extract_people(normalized)
        keyword_set = set(self.business_keywords)
        keywords = [token for token in jieba.lcut(normalized) if token in keyword_set]
        return ExtractedInfo(
            event_type=event_type,
            event_time=time_match.group(0) if time_match else None,
            event_location=location_match.group(1).strip() if location_match else None,
            people=people,
            keywords=list(dict.fromkeys(keywords)),
        )

    def _extract_people(self, text: str) -> list[Person]:
        found: dict[str, Person] = {}
        matched_organizations: set[str] = set()
        for entry in self.entities:
            person_token = next(
                (token for token in (entry.person_name, entry.person_alias) if token and token in text), None
            )
            org_token = next(
                (
                    token
                    for token in (entry.organization_name, entry.organization_alias)
                    if token and token in text
                ),
                None,
            )
            if person_token:
                found[entry.person_name] = Person(
                    name=entry.person_name,
                    organization=entry.organization_name if org_token else None,
                    title=self._title_near_person(text, person_token),
                )
                if org_token:
                    matched_organizations.add(entry.organization_name)

        for entry in self.entities:
            org_token = next(
                (
                    token
                    for token in (entry.organization_name, entry.organization_alias)
                    if token and token in text
                ),
                None,
            )
            if org_token and entry.organization_name not in matched_organizations:
                found.setdefault(
                    f"org:{entry.organization_name}",
                    Person(organization=entry.organization_name),
                )
                matched_organizations.add(entry.organization_name)

        sentences = [part for part in SENTENCE_SPLIT.split(text) if part]
        for sentence in sentences:
            names = []
            known_names = {entry.person_name for entry in self.entities if entry.person_name}
            for candidate in PERSON_PATTERN.findall(sentence):
                for connector in ("的", "和", "与", "及", "兼"):
                    if connector in candidate:
                        candidate = candidate.rsplit(connector, 1)[-1]
                if not 2 <= len(candidate) <= 4 or candidate in known_names:
                    continue
                if any(word in candidate for word in ("董事", "总裁", "经理", "集团", "公司")):
                    continue
                names.append(candidate)
            orgs = ORG_PATTERN.findall(sentence)
            title = next((item for item in self.titles if item in sentence), None)
            for index, name in enumerate(names):
                if name in found:
                    current = found[name]
                    found[name] = Person(
                        name=current.name or name,
                        organization=current.organization or (orgs[0] if orgs else None),
                        title=current.title or title,
                    )
                else:
                    found[name] = Person(
                        name=name,
                        organization=orgs[min(index, len(orgs) - 1)] if orgs else None,
                        title=title,
                    )

        if not found:
            orgs = ORG_PATTERN.findall(text)
            for org in dict.fromkeys(orgs):
                found[f"org:{org}"] = Person(organization=org)
        return list(found.values())

    def _title_near_person(self, text: str, person_token: str | None) -> str | None:
        if not person_token:
            return None
        escaped_person = re.escape(person_token)
        for title in self.titles:
            escaped_title = re.escape(title)
            if re.search(
                rf"(?:{escaped_person}.{{0,2}}{escaped_title}|{escaped_title}.{{0,2}}{escaped_person})",
                text,
            ):
                return title
        return None

    def extract_public_claims(
        self, pages: list[WebPage], extracted: ExtractedInfo
    ) -> list[PublicClaim]:
        person_names = list(
            dict.fromkeys(person.name for person in extracted.people if person.name)
        )
        organization_names = list(
            dict.fromkeys(person.organization for person in extracted.people if person.organization)
        )
        subjects = person_names or organization_names
        candidates: list[tuple[int, int, PublicClaim]] = []
        for page in pages:
            plain = BeautifulSoup(page.raw_content, "html.parser").get_text(" ")[:20_000]
            for sentence in SENTENCE_SPLIT.split(plain):
                sentence = re.sub(r"\s+", " ", sentence).strip()
                if not 10 <= len(sentence) <= 300:
                    continue
                matched_keywords = [word for word in self.business_keywords if word in sentence]
                if not matched_keywords:
                    continue
                for subject in subjects:
                    if subject not in sentence:
                        continue
                    score = 2 + len(matched_keywords)
                    candidates.append(
                        (
                            -score,
                            page.rank,
                            PublicClaim(
                                subject=subject,
                                claim=sentence,
                                source_title=page.title,
                                source_url=page.url,
                                matched_keywords=matched_keywords,
                            ),
                        )
                    )

        candidates.sort(key=lambda item: (item[0], item[1]))
        output: list[PublicClaim] = []
        per_subject: dict[str, int] = {}
        seen_sentences: set[str] = set()
        for _, _, claim in candidates:
            if claim.claim in seen_sentences or per_subject.get(claim.subject, 0) >= 3:
                continue
            seen_sentences.add(claim.claim)
            per_subject[claim.subject] = per_subject.get(claim.subject, 0) + 1
            output.append(claim)
        return output
