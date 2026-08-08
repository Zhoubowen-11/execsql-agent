"""Parameterized FSQ Shanghai SFT case families.

The factory contains no expected database values. Every expected result and final
answer is produced later from a real read-only SQLite execution.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SFTCase:
    """One parameterized training task before tool execution."""

    case_id: str
    template_family: str
    difficulty: str
    question: str
    sql: str
    inspect_tables: list[str]
    answer: dict[str, str]
    parameter_signature: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


FAMILY_CODES = {
    "district_alias_mapping": "a",
    "null_empty_filter": "b",
    "stable_top_k": "c",
    "multi_category": "d",
    "primary_category": "e",
    "contact_status": "f",
    "date_and_year": "g",
    "aggregation_shape": "h",
    "coordinate_semantics": "i",
    "quality_flags": "j",
}

DISTRICTS = [
    ("浦东新区", "Pudong"),
    ("徐汇区", "Xuhui"),
    ("黄浦区", "Huangpu"),
    ("静安区", "Jing'an"),
    ("长宁区", "Changning"),
]


def _literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _present(field: str) -> str:
    return f"{field} IS NOT NULL AND TRIM({field}) <> ''"


def _missing(field: str) -> str:
    return f"({field} IS NULL OR TRIM({field}) = '')"


def _scalar(template: str) -> dict[str, str]:
    return {"kind": "scalar", "template": template}


def _rows(template: str, row_template: str) -> dict[str, str]:
    return {"kind": "rows", "template": template, "row_template": row_template}


def _make(
    family: str,
    index: int,
    *,
    difficulty: str,
    question: str,
    sql: str,
    tables: list[str],
    answer: dict[str, str],
    parameters: dict[str, object],
) -> SFTCase:
    signature_payload = json.dumps(
        {"family": family, "parameters": parameters},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    signature = hashlib.sha256(signature_payload.encode("utf-8")).hexdigest()[:20]
    return SFTCase(
        case_id=f"sft_v1_{FAMILY_CODES[family]}_{index:03d}",
        template_family=family,
        difficulty=difficulty,
        question=question,
        sql=sql,
        inspect_tables=tables,
        answer=answer,
        parameter_signature=signature,
    )


def district_alias_mapping() -> list[SFTCase]:
    cases: list[SFTCase] = []
    index = 0
    for district_cn, district_en in DISTRICTS:
        district = _literal(district_en)
        specs = [
            (
                "total",
                f"{district_cn}目前收录了多少个地点？",
                f"SELECT COUNT(*) AS place_count FROM places WHERE canonical_district = {district}",
                f"{district_cn}共收录 {{place_count}} 个地点。",
                "easy",
                ["places"],
            ),
            (
                "cn_places",
                f"{district_cn}里国家标注为中国的地点有多少个？",
                f"SELECT COUNT(*) AS place_count FROM places "
                f"WHERE canonical_district = {district} AND country = 'CN'",
                f"{district_cn}国家标注为中国的地点共有 {{place_count}} 个。",
                "easy",
                ["places"],
            ),
            (
                "no_closed_date",
                f"{district_cn}中没有填写关闭日期的地点共有多少个？",
                f"SELECT COUNT(*) AS place_count FROM places "
                f"WHERE canonical_district = {district} AND {_missing('date_closed')}",
                f"{district_cn}没有填写关闭日期的地点共有 {{place_count}} 个。",
                "medium",
                ["places"],
            ),
            (
                "coffee_shop",
                f"{district_cn}被归为 Coffee Shop 的地点一共有多少个？",
                "SELECT COUNT(DISTINCT p.fsq_place_id) AS place_count "
                "FROM places p JOIN place_categories pc ON pc.fsq_place_id = p.fsq_place_id "
                "JOIN categories c ON c.category_id = pc.category_id "
                f"WHERE p.canonical_district = {district} "
                "AND c.category_name = 'Coffee Shop'",
                f"{district_cn}被归为 Coffee Shop 的地点共有 {{place_count}} 个。",
                "medium",
                ["places", "place_categories", "categories"],
            ),
        ]
        for variant, question, sql, answer, difficulty, tables in specs:
            index += 1
            cases.append(
                _make(
                    "district_alias_mapping",
                    index,
                    difficulty=difficulty,
                    question=question,
                    sql=sql,
                    tables=tables,
                    answer=_scalar(answer),
                    parameters={"district": district_en, "variant": variant},
                )
            )
    return cases


def null_empty_filter() -> list[SFTCase]:
    cases: list[SFTCase] = []
    fields = [
        ("tel", "联系电话"),
        ("website", "官方网站"),
        ("postcode", "邮政编码"),
        ("locality", "原始城市标注"),
        ("email", "邮箱"),
    ]
    for offset, (field, label) in enumerate(fields):
        district_cn, district_en = DISTRICTS[offset]
        specs = [
            (
                "present_all",
                f"有填写{label}的地点共有多少个？",
                f"SELECT COUNT(*) AS place_count FROM places WHERE {_present(field)}",
                f"有填写{label}的地点共有 {{place_count}} 个。",
            ),
            (
                "missing_all",
                f"{label}为空或没有记录的地点有多少个？",
                f"SELECT COUNT(*) AS place_count FROM places WHERE {_missing(field)}",
                f"{label}为空或没有记录的地点共有 {{place_count}} 个。",
            ),
            (
                "present_district",
                f"{district_cn}有填写{label}的地点有多少个？",
                f"SELECT COUNT(*) AS place_count FROM places "
                f"WHERE canonical_district = {_literal(district_en)} "
                f"AND {_present(field)}",
                f"{district_cn}有填写{label}的地点共有 {{place_count}} 个。",
            ),
            (
                "missing_district",
                f"{district_cn}缺少{label}的地点有多少个？空白内容也算缺少。",
                f"SELECT COUNT(*) AS place_count FROM places "
                f"WHERE canonical_district = {_literal(district_en)} "
                f"AND {_missing(field)}",
                f"{district_cn}缺少{label}的地点共有 {{place_count}} 个。",
            ),
        ]
        for variant, question, sql, answer in specs:
            cases.append(
                _make(
                    "null_empty_filter",
                    len(cases) + 1,
                    difficulty="easy" if "all" in variant else "medium",
                    question=question,
                    sql=sql,
                    tables=["places"],
                    answer=_scalar(answer),
                    parameters={"field": field, "variant": variant, "district": district_en},
                )
            )
    return cases


def stable_top_k() -> list[SFTCase]:
    cases: list[SFTCase] = []
    district_specs = [
        (3, "全部地点", "1 = 1"),
        (5, "留有电话的地点", _present("tel")),
        (10, "留有官网的地点", _present("website")),
        (3, "留有邮箱的地点", _present("email")),
        (5, "有邮政编码的地点", _present("postcode")),
        (10, "缺少电话的地点", _missing("tel")),
        (3, "缺少官网的地点", _missing("website")),
        (5, "有关闭日期的地点", _present("date_closed")),
        (10, "2025 年创建的地点", "substr(date_created, 1, 4) = '2025'"),
        (3, "2026 年刷新的地点", "substr(date_refreshed, 1, 4) = '2026'"),
    ]
    for k, label, condition in district_specs:
        sql = (
            "SELECT canonical_district, COUNT(*) AS place_count FROM places "
            "WHERE canonical_district IS NOT NULL "
            f"AND {condition} GROUP BY canonical_district "
            "ORDER BY place_count DESC, canonical_district ASC "
            f"LIMIT {k}"
        )
        cases.append(
            _make(
                "stable_top_k",
                len(cases) + 1,
                difficulty="medium",
                question=f"按{label}数量看，排名前 {k} 的行政区有哪些？并列时按区名排序。",
                sql=sql,
                tables=["places"],
                answer=_rows(
                    f"排名前 {k} 的行政区是：{{rows}}。",
                    "{rank}. {canonical_district}（{place_count} 个）",
                ),
                parameters={"shape": "district", "k": k, "condition": condition},
            )
        )

    category_conditions = [
        (5, "全部分类关联", "1 = 1"),
        (10, "推断为首项分类的关联", "pc.is_primary = 1"),
        (3, "留有电话的地点", _present("p.tel")),
        (5, "留有官网的地点", _present("p.website")),
        (10, "没有关闭日期的地点", _missing("p.date_closed")),
    ]
    for k, label, condition in category_conditions:
        sql = (
            "SELECT c.category_name, COUNT(DISTINCT p.fsq_place_id) AS place_count "
            "FROM places p JOIN place_categories pc ON pc.fsq_place_id = p.fsq_place_id "
            "JOIN categories c ON c.category_id = pc.category_id "
            f"WHERE {condition} GROUP BY c.category_id, c.category_name "
            "ORDER BY place_count DESC, c.category_name ASC "
            f"LIMIT {k}"
        )
        cases.append(
            _make(
                "stable_top_k",
                len(cases) + 1,
                difficulty="hard",
                question=(
                    f"按{label}统计，覆盖地点最多的前 {k} 个类型是什么？同数时按类型名称排序。"
                ),
                sql=sql,
                tables=["places", "place_categories", "categories"],
                answer=_rows(
                    f"覆盖地点最多的前 {k} 个类型是：{{rows}}。",
                    "{rank}. {category_name}（{place_count} 个地点）",
                ),
                parameters={"shape": "category", "k": k, "condition": condition},
            )
        )

    locality_conditions = [
        (3, "全部地点", "1 = 1"),
        (5, "留有电话的地点", _present("tel")),
        (10, "留有官网的地点", _present("website")),
        (3, "有关闭日期的地点", _present("date_closed")),
        (5, "没有邮政编码的地点", _missing("postcode")),
    ]
    for k, label, condition in locality_conditions:
        sql = (
            "SELECT locality, COUNT(*) AS place_count FROM places "
            f"WHERE {_present('locality')} AND {condition} GROUP BY locality "
            "ORDER BY place_count DESC, locality ASC "
            f"LIMIT {k}"
        )
        cases.append(
            _make(
                "stable_top_k",
                len(cases) + 1,
                difficulty="medium",
                question=(
                    f"原始城市标注中，按{label}数量排在前 {k} 的标注是什么？同数时按名称排序。"
                ),
                sql=sql,
                tables=["places"],
                answer=_rows(
                    f"排名前 {k} 的原始城市标注是：{{rows}}。",
                    "{rank}. {locality}（{place_count} 个）",
                ),
                parameters={"shape": "locality", "k": k, "condition": condition},
            )
        )
    return cases


def multi_category() -> list[SFTCase]:
    cases: list[SFTCase] = []
    for threshold in range(1, 6):
        sql = (
            "SELECT COUNT(*) AS place_count FROM ("
            "SELECT fsq_place_id FROM place_categories GROUP BY fsq_place_id "
            f"HAVING COUNT(*) > {threshold}) AS multi_category_places"
        )
        cases.append(
            _make(
                "multi_category",
                len(cases) + 1,
                difficulty="medium",
                question=f"至少关联 {threshold + 1} 个分类的地点一共有多少个？",
                sql=sql,
                tables=["place_categories"],
                answer=_scalar(f"至少关联 {threshold + 1} 个分类的地点共有 {{place_count}} 个。"),
                parameters={"scope": "all", "threshold": threshold},
            )
        )

    for district_cn, district_en in DISTRICTS:
        for threshold in (1, 2):
            sql = (
                "SELECT COUNT(*) AS place_count FROM ("
                "SELECT p.fsq_place_id FROM places p "
                "JOIN place_categories pc ON pc.fsq_place_id = p.fsq_place_id "
                f"WHERE p.canonical_district = {_literal(district_en)} "
                "GROUP BY p.fsq_place_id "
                f"HAVING COUNT(*) > {threshold}) AS district_multi_category_places"
            )
            cases.append(
                _make(
                    "multi_category",
                    len(cases) + 1,
                    difficulty="hard",
                    question=f"{district_cn}至少关联 {threshold + 1} 个分类的地点有多少个？",
                    sql=sql,
                    tables=["places", "place_categories"],
                    answer=_scalar(
                        f"{district_cn}至少关联 {threshold + 1} 个分类的地点"
                        "共有 {place_count} 个。"
                    ),
                    parameters={"scope": district_en, "threshold": threshold},
                )
            )

    ranking_specs = [(1, 3), (1, 5), (2, 5), (2, 10), (3, 5)]
    for threshold, k in ranking_specs:
        sql = (
            "SELECT p.fsq_place_id, p.name, COUNT(*) AS category_count "
            "FROM places p JOIN place_categories pc ON pc.fsq_place_id = p.fsq_place_id "
            "GROUP BY p.fsq_place_id, p.name "
            f"HAVING COUNT(*) > {threshold} "
            "ORDER BY category_count DESC, p.fsq_place_id ASC "
            f"LIMIT {k}"
        )
        cases.append(
            _make(
                "multi_category",
                len(cases) + 1,
                difficulty="hard",
                question=(
                    f"关联分类最多的前 {k} 个地点是哪些？"
                    f"只看至少有 {threshold + 1} 个分类的地点，"
                    "同数时按地点编号排序。"
                ),
                sql=sql,
                tables=["places", "place_categories"],
                answer=_rows(
                    f"关联分类最多的前 {k} 个地点是：{{rows}}。",
                    "{rank}. {name}（编号 {fsq_place_id}，{category_count} 个分类）",
                ),
                parameters={"scope": "ranking", "threshold": threshold, "k": k},
            )
        )
    return cases


def primary_category() -> list[SFTCase]:
    cases: list[SFTCase] = []
    for district_cn, district_en in DISTRICTS:
        base_join = "FROM places p JOIN place_categories pc ON pc.fsq_place_id = p.fsq_place_id "
        sql = (
            "SELECT COUNT(DISTINCT p.fsq_place_id) AS place_count "
            + base_join
            + f"WHERE p.canonical_district = {_literal(district_en)} AND pc.is_primary = 1"
        )
        cases.append(
            _make(
                "primary_category",
                len(cases) + 1,
                difficulty="medium",
                question=f"{district_cn}有首项分类标记的地点共有多少个？这里的首项只代表源数组第一项。",
                sql=sql,
                tables=["places", "place_categories"],
                answer=_scalar(f"{district_cn}有首项分类标记的地点共有 {{place_count}} 个。"),
                parameters={"shape": "district_count", "district": district_en},
            )
        )

    for district_cn, district_en in DISTRICTS:
        sql = (
            "SELECT c.category_name, COUNT(DISTINCT p.fsq_place_id) AS place_count "
            "FROM places p JOIN place_categories pc ON pc.fsq_place_id = p.fsq_place_id "
            "JOIN categories c ON c.category_id = pc.category_id "
            f"WHERE p.canonical_district = {_literal(district_en)} AND pc.is_primary = 1 "
            "GROUP BY c.category_id, c.category_name "
            "ORDER BY place_count DESC, c.category_name ASC LIMIT 5"
        )
        cases.append(
            _make(
                "primary_category",
                len(cases) + 1,
                difficulty="hard",
                question=f"{district_cn}按首项分类标记统计，覆盖地点最多的五个类型是什么？同数时按类型名称排序。",
                sql=sql,
                tables=["places", "place_categories", "categories"],
                answer=_rows(
                    f"{district_cn}首项分类覆盖最多的五个类型是：{{rows}}。",
                    "{rank}. {category_name}（{place_count} 个地点）",
                ),
                parameters={"shape": "district_top", "district": district_en},
            )
        )

    contact_conditions = [
        ("有电话", _present("p.tel")),
        ("有官网", _present("p.website")),
        ("电话和官网都有", f"{_present('p.tel')} AND {_present('p.website')}"),
        ("电话和官网都缺少", f"{_missing('p.tel')} AND {_missing('p.website')}"),
        ("有关闭日期", _present("p.date_closed")),
    ]
    for label, condition in contact_conditions:
        sql = (
            "SELECT c.category_name, COUNT(DISTINCT p.fsq_place_id) AS place_count "
            "FROM places p JOIN place_categories pc ON pc.fsq_place_id = p.fsq_place_id "
            "JOIN categories c ON c.category_id = pc.category_id "
            f"WHERE pc.is_primary = 1 AND {condition} "
            "GROUP BY c.category_id, c.category_name "
            "ORDER BY place_count DESC, c.category_name ASC LIMIT 5"
        )
        cases.append(
            _make(
                "primary_category",
                len(cases) + 1,
                difficulty="hard",
                question=f"在{label}的地点中，首项分类覆盖最多的五个类型是什么？同数时按类型名称排序。",
                sql=sql,
                tables=["places", "place_categories", "categories"],
                answer=_rows(
                    f"{label}的地点中，首项分类覆盖最多的五个类型是：{{rows}}。",
                    "{rank}. {category_name}（{place_count} 个地点）",
                ),
                parameters={"shape": "contact_top", "condition": label},
            )
        )

    for level in range(1, 6):
        sql = (
            "SELECT COUNT(DISTINCT pc.fsq_place_id) AS place_count "
            "FROM place_categories pc JOIN categories c ON c.category_id = pc.category_id "
            f"WHERE pc.is_primary = 1 AND c.category_level = {level}"
        )
        cases.append(
            _make(
                "primary_category",
                len(cases) + 1,
                difficulty="medium",
                question=f"首项分类处在第 {level} 层的地点有多少个？",
                sql=sql,
                tables=["place_categories", "categories"],
                answer=_scalar(f"首项分类处在第 {level} 层的地点共有 {{place_count}} 个。"),
                parameters={"shape": "level_count", "level": level},
            )
        )
    return cases


def contact_status() -> list[SFTCase]:
    cases: list[SFTCase] = []
    tel_present = _present("tel")
    web_present = _present("website")
    tel_missing = _missing("tel")
    web_missing = _missing("website")
    statuses = [
        ("both", "电话和官网都齐全", f"{tel_present} AND {web_present}"),
        (
            "one",
            "电话和官网恰好有一项",
            f"(({tel_present}) AND {web_missing}) OR ({tel_missing} AND ({web_present}))",
        ),
        ("neither", "电话和官网都没有", f"{tel_missing} AND {web_missing}"),
    ]
    for district_cn, district_en in DISTRICTS:
        for status, label, condition in statuses:
            sql = (
                "SELECT COUNT(*) AS place_count FROM places "
                f"WHERE canonical_district = {_literal(district_en)} AND ({condition})"
            )
            cases.append(
                _make(
                    "contact_status",
                    len(cases) + 1,
                    difficulty="medium",
                    question=f"{district_cn}{label}的地点有多少个？",
                    sql=sql,
                    tables=["places"],
                    answer=_scalar(f"{district_cn}{label}的地点共有 {{place_count}} 个。"),
                    parameters={"scope": district_en, "status": status},
                )
            )

    for status, label, condition in statuses:
        sql = f"SELECT COUNT(*) AS place_count FROM places WHERE {condition}"
        cases.append(
            _make(
                "contact_status",
                len(cases) + 1,
                difficulty="easy",
                question=f"全库中{label}的地点有多少个？",
                sql=sql,
                tables=["places"],
                answer=_scalar(f"全库中{label}的地点共有 {{place_count}} 个。"),
                parameters={"scope": "all", "status": status},
            )
        )

    exclusive_specs = [
        ("tel_only", "只有电话但没有官网", f"{tel_present} AND {web_missing}"),
        ("web_only", "只有官网但没有电话", f"{web_present} AND {tel_missing}"),
    ]
    for status, label, condition in exclusive_specs:
        cases.append(
            _make(
                "contact_status",
                len(cases) + 1,
                difficulty="medium",
                question=f"全库中{label}的地点有多少个？",
                sql=f"SELECT COUNT(*) AS place_count FROM places WHERE {condition}",
                tables=["places"],
                answer=_scalar(f"全库中{label}的地点共有 {{place_count}} 个。"),
                parameters={"scope": "all", "status": status},
            )
        )
    return cases


def date_and_year() -> list[SFTCase]:
    cases: list[SFTCase] = []
    for district_cn, district_en in DISTRICTS:
        sql = (
            "SELECT COUNT(*) AS place_count FROM places "
            f"WHERE canonical_district = {_literal(district_en)} "
            f"AND {_present('date_closed')}"
        )
        cases.append(
            _make(
                "date_and_year",
                len(cases) + 1,
                difficulty="easy",
                question=f"{district_cn}明确记录了关闭日期的地点有多少个？",
                sql=sql,
                tables=["places"],
                answer=_scalar(f"{district_cn}明确记录关闭日期的地点共有 {{place_count}} 个。"),
                parameters={"shape": "district_closed", "district": district_en},
            )
        )

    for k in (3, 5, 10):
        sql = (
            "SELECT substr(date_closed, 1, 4) AS closed_year, COUNT(*) AS place_count "
            f"FROM places WHERE {_present('date_closed')} "
            "GROUP BY substr(date_closed, 1, 4) "
            "ORDER BY place_count DESC, closed_year DESC "
            f"LIMIT {k}"
        )
        cases.append(
            _make(
                "date_and_year",
                len(cases) + 1,
                difficulty="medium",
                question=f"关闭地点最多的前 {k} 个年份是哪几年？同数时较新的年份优先。",
                sql=sql,
                tables=["places"],
                answer=_rows(
                    f"关闭地点最多的前 {k} 个年份是：{{rows}}。",
                    "{rank}. {closed_year} 年（{place_count} 个）",
                ),
                parameters={"shape": "closed_year_top", "k": k},
            )
        )

    for k in (3, 5, 10):
        sql = (
            "SELECT fsq_place_id, name, date_closed FROM places "
            f"WHERE {_present('date_closed')} "
            "ORDER BY date_closed DESC, fsq_place_id ASC "
            f"LIMIT {k}"
        )
        cases.append(
            _make(
                "date_and_year",
                len(cases) + 1,
                difficulty="medium",
                question=f"关闭日期最近的 {k} 个地点是哪些？日期相同时按地点编号排序。",
                sql=sql,
                tables=["places"],
                answer=_rows(
                    f"关闭日期最近的 {k} 个地点是：{{rows}}。",
                    "{rank}. {name}（{date_closed}，编号 {fsq_place_id}）",
                ),
                parameters={"shape": "recent_closed", "k": k},
            )
        )

    for year in (2024, 2025, 2026):
        sql = (
            f"SELECT COUNT(*) AS place_count FROM places WHERE substr(date_closed, 1, 4) = '{year}'"
        )
        cases.append(
            _make(
                "date_and_year",
                len(cases) + 1,
                difficulty="easy",
                question=f"关闭日期落在 {year} 年的地点有多少个？",
                sql=sql,
                tables=["places"],
                answer=_scalar(f"关闭日期落在 {year} 年的地点共有 {{place_count}} 个。"),
                parameters={"shape": "closed_year_count", "year": year},
            )
        )

    date_fields = [
        ("date_closed", "关闭日期"),
        ("date_created", "创建日期"),
        ("date_refreshed", "刷新日期"),
    ]
    for field, label in date_fields:
        cases.append(
            _make(
                "date_and_year",
                len(cases) + 1,
                difficulty="easy",
                question=f"没有填写{label}的地点有多少个？",
                sql=f"SELECT COUNT(*) AS place_count FROM places WHERE {_missing(field)}",
                tables=["places"],
                answer=_scalar(f"没有填写{label}的地点共有 {{place_count}} 个。"),
                parameters={"shape": "missing_date", "field": field},
            )
        )

    for k in (3, 5, 10):
        sql = (
            "SELECT substr(date_created, 1, 4) AS created_year, COUNT(*) AS place_count "
            f"FROM places WHERE {_present('date_created')} "
            "GROUP BY substr(date_created, 1, 4) "
            "ORDER BY place_count DESC, created_year DESC "
            f"LIMIT {k}"
        )
        cases.append(
            _make(
                "date_and_year",
                len(cases) + 1,
                difficulty="medium",
                question=f"新建地点数量最多的前 {k} 个年份是哪几年？同数时较新的年份优先。",
                sql=sql,
                tables=["places"],
                answer=_rows(
                    f"新建地点最多的前 {k} 个年份是：{{rows}}。",
                    "{rank}. {created_year} 年（{place_count} 个）",
                ),
                parameters={"shape": "created_year_top", "k": k},
            )
        )
    return cases


def aggregation_shape() -> list[SFTCase]:
    cases: list[SFTCase] = []
    scalar_specs = [
        (
            "distinct_countries",
            "地点数据里一共出现了多少种国家标注？",
            "SELECT COUNT(DISTINCT country) AS value_count FROM places",
            "国家标注共有 {value_count} 种。",
        ),
        (
            "mapped_districts",
            "能明确识别出的行政区一共有多少个？",
            "SELECT COUNT(DISTINCT canonical_district) AS value_count FROM places "
            "WHERE canonical_district IS NOT NULL",
            "能明确识别出的行政区共有 {value_count} 个。",
        ),
        (
            "average_coordinates",
            "所有地点的平均纬度和平均经度分别是多少？",
            "SELECT ROUND(AVG(latitude), 6) AS avg_latitude, "
            "ROUND(AVG(longitude), 6) AS avg_longitude FROM places",
            None,
        ),
        (
            "created_range",
            "地点创建日期最早和最晚分别是哪一天？",
            "SELECT MIN(date_created) AS earliest_date, MAX(date_created) AS latest_date "
            f"FROM places WHERE {_present('date_created')}",
            None,
        ),
        (
            "total_and_locality",
            "地点总数和不同原始城市标注数分别是多少？",
            "SELECT COUNT(*) AS total_count, COUNT(DISTINCT locality) AS locality_count "
            "FROM places",
            None,
        ),
    ]
    for key, question, sql, scalar_template in scalar_specs:
        answer = (
            _scalar(scalar_template)
            if scalar_template is not None
            else _rows("查询结果是：{rows}。", "{rank}. {row_summary}")
        )
        if scalar_template is None:
            if key == "average_coordinates":
                answer = _rows(
                    "平均坐标为：{rows}。",
                    "纬度 {avg_latitude}，经度 {avg_longitude}",
                )
            elif key == "created_range":
                answer = _rows(
                    "创建日期范围为：{rows}。",
                    "最早 {earliest_date}，最晚 {latest_date}",
                )
            else:
                answer = _rows(
                    "汇总结果为：{rows}。",
                    "地点 {total_count} 个，原始城市标注 {locality_count} 种",
                )
        cases.append(
            _make(
                "aggregation_shape",
                len(cases) + 1,
                difficulty="medium",
                question=question,
                sql=sql,
                tables=["places"],
                answer=answer,
                parameters={"shape": key},
            )
        )

    conditions = [
        ("全部地点", "1 = 1"),
        ("有电话的地点", _present("tel")),
        ("有官网的地点", _present("website")),
        ("没有关闭日期的地点", _missing("date_closed")),
        ("有邮政编码的地点", _present("postcode")),
    ]
    for label, condition in conditions:
        sql = (
            "SELECT country, canonical_district, COUNT(*) AS place_count FROM places "
            f"WHERE canonical_district IS NOT NULL AND {condition} "
            "GROUP BY country, canonical_district "
            "ORDER BY country ASC, canonical_district ASC"
        )
        cases.append(
            _make(
                "aggregation_shape",
                len(cases) + 1,
                difficulty="medium",
                question=f"按国家标注和可识别行政区展开，{label}分别有多少个？",
                sql=sql,
                tables=["places"],
                answer=_rows(
                    f"{label}按国家和行政区展开为：{{rows}}。",
                    "{country} / {canonical_district}：{place_count} 个",
                ),
                parameters={"shape": "country_district", "condition": label},
            )
        )

    category_conditions = [
        ("全部分类关联", "1 = 1"),
        ("首项分类关联", "pc.is_primary = 1"),
        ("有电话的地点", _present("p.tel")),
        ("有官网的地点", _present("p.website")),
        ("没有关闭日期的地点", _missing("p.date_closed")),
    ]
    for label, condition in category_conditions:
        sql = (
            "SELECT c.category_level, pc.is_primary, "
            "COUNT(DISTINCT p.fsq_place_id) AS place_count "
            "FROM places p JOIN place_categories pc ON pc.fsq_place_id = p.fsq_place_id "
            "JOIN categories c ON c.category_id = pc.category_id "
            f"WHERE {condition} GROUP BY c.category_level, pc.is_primary "
            "ORDER BY c.category_level ASC, pc.is_primary DESC"
        )
        cases.append(
            _make(
                "aggregation_shape",
                len(cases) + 1,
                difficulty="hard",
                question=f"把{label}按分类层级和是否为首项标记展开，各组合覆盖多少个地点？",
                sql=sql,
                tables=["places", "place_categories", "categories"],
                answer=_rows(
                    f"{label}按层级和首项标记展开为：{{rows}}。",
                    "层级 {category_level}，首项标记 {is_primary}：{place_count} 个地点",
                ),
                parameters={"shape": "level_primary", "condition": label},
            )
        )

    for district_cn, district_en in DISTRICTS:
        sql = (
            "SELECT CASE "
            f"WHEN {_present('tel')} AND {_present('website')} THEN 'both' "
            f"WHEN {_present('tel')} OR {_present('website')} THEN 'one' "
            "ELSE 'neither' END AS contact_status, "
            "country, COUNT(*) AS place_count FROM places "
            f"WHERE canonical_district = {_literal(district_en)} "
            "GROUP BY contact_status, country "
            "ORDER BY contact_status ASC, country ASC"
        )
        cases.append(
            _make(
                "aggregation_shape",
                len(cases) + 1,
                difficulty="hard",
                question=f"{district_cn}按联系方式完整程度和国家标注展开，各组合有多少个地点？",
                sql=sql,
                tables=["places"],
                answer=_rows(
                    f"{district_cn}的联系方式组合为：{{rows}}。",
                    "{contact_status} / {country}：{place_count} 个",
                ),
                parameters={"shape": "contact_country", "district": district_en},
            )
        )
    return cases


def coordinate_semantics() -> list[SFTCase]:
    cases: list[SFTCase] = []
    scalar_specs = [
        (
            "west",
            "最西侧地点的经度是多少？",
            "MIN(longitude)",
            "west_longitude",
            "最西侧经度为 {west_longitude}。",
        ),
        (
            "east",
            "最东侧地点的经度是多少？",
            "MAX(longitude)",
            "east_longitude",
            "最东侧经度为 {east_longitude}。",
        ),
        (
            "south",
            "最南侧地点的纬度是多少？",
            "MIN(latitude)",
            "south_latitude",
            "最南侧纬度为 {south_latitude}。",
        ),
        (
            "north",
            "最北侧地点的纬度是多少？",
            "MAX(latitude)",
            "north_latitude",
            "最北侧纬度为 {north_latitude}。",
        ),
    ]
    for direction, question, expression, alias, template in scalar_specs:
        cases.append(
            _make(
                "coordinate_semantics",
                len(cases) + 1,
                difficulty="easy",
                question=question,
                sql=f"SELECT {expression} AS {alias} FROM places",
                tables=["places"],
                answer=_scalar(template),
                parameters={"shape": "scalar_extreme", "direction": direction},
            )
        )

    row_specs = [
        ("west", "最西侧的地点是哪一个？", "longitude", "ASC"),
        ("east", "最东侧的地点是哪一个？", "longitude", "DESC"),
        ("south", "最南侧的地点是哪一个？", "latitude", "ASC"),
        ("north", "最北侧的地点是哪一个？", "latitude", "DESC"),
    ]
    for direction, question, coordinate, order in row_specs:
        sql = (
            "SELECT fsq_place_id, name, latitude, longitude FROM places "
            f"ORDER BY {coordinate} {order}, fsq_place_id ASC LIMIT 1"
        )
        cases.append(
            _make(
                "coordinate_semantics",
                len(cases) + 1,
                difficulty="medium",
                question=question + "坐标相同时按地点编号确定唯一结果。",
                sql=sql,
                tables=["places"],
                answer=_rows(
                    "对应地点是：{rows}。",
                    "{name}（编号 {fsq_place_id}，纬度 {latitude}，经度 {longitude}）",
                ),
                parameters={"shape": "place_extreme", "direction": direction},
            )
        )

    for district_cn, district_en in DISTRICTS:
        sql = (
            "SELECT MIN(longitude) AS west_longitude, MAX(longitude) AS east_longitude, "
            "MIN(latitude) AS south_latitude, MAX(latitude) AS north_latitude "
            f"FROM places WHERE canonical_district = {_literal(district_en)}"
        )
        cases.append(
            _make(
                "coordinate_semantics",
                len(cases) + 1,
                difficulty="medium",
                question=f"{district_cn}地点的最西、最东、最南和最北坐标边界分别是多少？",
                sql=sql,
                tables=["places"],
                answer=_rows(
                    f"{district_cn}的坐标边界为：{{rows}}。",
                    "西 {west_longitude}，东 {east_longitude}，"
                    "南 {south_latitude}，北 {north_latitude}",
                ),
                parameters={"shape": "district_bounds", "district": district_en},
            )
        )

    threshold_specs = [
        ("west_of", "经度小于 121.40 的地点有多少个？", "longitude < 121.40"),
        ("east_of", "经度大于等于 121.60 的地点有多少个？", "longitude >= 121.60"),
        ("south_of", "纬度低于 31.10 的地点有多少个？", "latitude < 31.10"),
        ("north_of", "纬度高于等于 31.30 的地点有多少个？", "latitude >= 31.30"),
    ]
    for key, question, condition in threshold_specs:
        cases.append(
            _make(
                "coordinate_semantics",
                len(cases) + 1,
                difficulty="easy",
                question=question,
                sql=f"SELECT COUNT(*) AS place_count FROM places WHERE {condition}",
                tables=["places"],
                answer=_scalar("符合该坐标条件的地点共有 {place_count} 个。"),
                parameters={"shape": "threshold", "condition": key},
            )
        )

    rectangles = [
        ("中心区域", 121.40, 121.50, 31.18, 31.25),
        ("东部矩形范围", 121.50, 121.70, 31.15, 31.35),
        ("西南矩形范围", 121.20, 121.45, 30.95, 31.20),
    ]
    for label, west, east, south, north in rectangles:
        sql = (
            "SELECT COUNT(*) AS place_count FROM places "
            f"WHERE longitude BETWEEN {west} AND {east} "
            f"AND latitude BETWEEN {south} AND {north}"
        )
        cases.append(
            _make(
                "coordinate_semantics",
                len(cases) + 1,
                difficulty="medium",
                question=(
                    f"在{label}（经度 {west} 到 {east}、纬度 {south} 到 {north}）内有多少个地点？"
                ),
                sql=sql,
                tables=["places"],
                answer=_scalar(f"{label}内共有 {{place_count}} 个地点。"),
                parameters={"shape": "rectangle", "bounds": [west, east, south, north]},
            )
        )
    return cases


def quality_flags() -> list[SFTCase]:
    cases: list[SFTCase] = []
    scalar_specs = [
        (
            "distinct_flagged",
            "至少带有一个未解决质量标记的地点有多少个？",
            "SELECT COUNT(DISTINCT fsq_place_id) AS place_count FROM place_unresolved_flags",
            "至少带有一个未解决质量标记的地点共有 {place_count} 个。",
        ),
        (
            "flag_links",
            "未解决质量标记与地点之间一共有多少条关联记录？",
            "SELECT COUNT(*) AS link_count FROM place_unresolved_flags",
            "未解决质量标记关联共有 {link_count} 条。",
        ),
    ]
    for key, question, sql, template in scalar_specs:
        cases.append(
            _make(
                "quality_flags",
                len(cases) + 1,
                difficulty="easy",
                question=question,
                sql=sql,
                tables=["place_unresolved_flags"],
                answer=_scalar(template),
                parameters={"shape": key},
            )
        )

    for k in (3, 5, 10):
        sql = (
            "SELECT flag, COUNT(DISTINCT fsq_place_id) AS place_count "
            "FROM place_unresolved_flags GROUP BY flag "
            "ORDER BY place_count DESC, flag ASC "
            f"LIMIT {k}"
        )
        cases.append(
            _make(
                "quality_flags",
                len(cases) + 1,
                difficulty="medium",
                question=f"影响地点最多的前 {k} 种未解决质量标记是什么？同数时按标记名称排序。",
                sql=sql,
                tables=["place_unresolved_flags"],
                answer=_rows(
                    f"影响地点最多的前 {k} 种质量标记是：{{rows}}。",
                    "{rank}. {flag}（{place_count} 个地点）",
                ),
                parameters={"shape": "flag_top", "k": k},
            )
        )

    for district_cn, district_en in DISTRICTS:
        sql = (
            "SELECT f.flag, COUNT(DISTINCT f.fsq_place_id) AS place_count "
            "FROM place_unresolved_flags f JOIN places p ON p.fsq_place_id = f.fsq_place_id "
            f"WHERE p.canonical_district = {_literal(district_en)} "
            "GROUP BY f.flag ORDER BY place_count DESC, f.flag ASC LIMIT 5"
        )
        cases.append(
            _make(
                "quality_flags",
                len(cases) + 1,
                difficulty="hard",
                question=f"{district_cn}影响地点最多的五种未解决质量标记是什么？同数时按标记名称排序。",
                sql=sql,
                tables=["places", "place_unresolved_flags"],
                answer=_rows(
                    f"{district_cn}影响地点最多的五种质量标记是：{{rows}}。",
                    "{rank}. {flag}（{place_count} 个地点）",
                ),
                parameters={"shape": "district_flag_top", "district": district_en},
            )
        )

    for district_cn, district_en in DISTRICTS:
        sql = (
            "SELECT COUNT(DISTINCT f.fsq_place_id) AS place_count "
            "FROM place_unresolved_flags f JOIN places p ON p.fsq_place_id = f.fsq_place_id "
            f"WHERE p.canonical_district = {_literal(district_en)}"
        )
        cases.append(
            _make(
                "quality_flags",
                len(cases) + 1,
                difficulty="medium",
                question=f"{district_cn}至少带有一种未解决质量标记的地点有多少个？",
                sql=sql,
                tables=["places", "place_unresolved_flags"],
                answer=_scalar(f"{district_cn}带有未解决质量标记的地点共有 {{place_count}} 个。"),
                parameters={"shape": "district_flagged", "district": district_en},
            )
        )

    for threshold in (1, 2):
        sql = (
            "SELECT COUNT(*) AS place_count FROM ("
            "SELECT fsq_place_id FROM place_unresolved_flags GROUP BY fsq_place_id "
            f"HAVING COUNT(*) > {threshold}) AS multi_flag_places"
        )
        cases.append(
            _make(
                "quality_flags",
                len(cases) + 1,
                difficulty="medium",
                question=f"同时带有至少 {threshold + 1} 种未解决质量标记的地点有多少个？",
                sql=sql,
                tables=["place_unresolved_flags"],
                answer=_scalar(
                    f"同时带有至少 {threshold + 1} 种未解决质量标记的地点共有 {{place_count}} 个。"
                ),
                parameters={"shape": "multi_flag", "threshold": threshold},
            )
        )

    for k in (5, 10):
        sql = (
            "SELECT p.canonical_district, "
            "COUNT(DISTINCT f.fsq_place_id) AS place_count "
            "FROM place_unresolved_flags f JOIN places p ON p.fsq_place_id = f.fsq_place_id "
            "WHERE p.canonical_district IS NOT NULL GROUP BY p.canonical_district "
            "ORDER BY place_count DESC, p.canonical_district ASC "
            f"LIMIT {k}"
        )
        cases.append(
            _make(
                "quality_flags",
                len(cases) + 1,
                difficulty="hard",
                question=f"带质量标记地点最多的前 {k} 个可识别行政区是哪些？同数时按区名排序。",
                sql=sql,
                tables=["places", "place_unresolved_flags"],
                answer=_rows(
                    f"带质量标记地点最多的前 {k} 个行政区是：{{rows}}。",
                    "{rank}. {canonical_district}（{place_count} 个地点）",
                ),
                parameters={"shape": "district_top", "k": k},
            )
        )

    sql = (
        "SELECT c.category_name, COUNT(DISTINCT f.fsq_place_id) AS place_count "
        "FROM place_unresolved_flags f "
        "JOIN place_categories pc ON pc.fsq_place_id = f.fsq_place_id "
        "JOIN categories c ON c.category_id = pc.category_id "
        "WHERE pc.is_primary = 1 GROUP BY c.category_id, c.category_name "
        "ORDER BY place_count DESC, c.category_name ASC LIMIT 5"
    )
    cases.append(
        _make(
            "quality_flags",
            len(cases) + 1,
            difficulty="hard",
            question="带未解决质量标记的地点中，首项分类覆盖最多的五个类型是什么？同数时按名称排序。",
            sql=sql,
            tables=["place_unresolved_flags", "place_categories", "categories"],
            answer=_rows(
                "带质量标记地点的首项分类前五名是：{rows}。",
                "{rank}. {category_name}（{place_count} 个地点）",
            ),
            parameters={"shape": "primary_category_top"},
        )
    )
    return cases


def generate_cases() -> list[SFTCase]:
    """Return exactly 20 unique parameter combinations per template family."""

    factories = [
        district_alias_mapping,
        null_empty_filter,
        stable_top_k,
        multi_category,
        primary_category,
        contact_status,
        date_and_year,
        aggregation_shape,
        coordinate_semantics,
        quality_flags,
    ]
    cases: list[SFTCase] = []
    for factory in factories:
        family_cases = factory()
        if len(family_cases) != 20:
            raise ValueError(f"{factory.__name__} generated {len(family_cases)}, expected 20")
        cases.extend(family_cases)
    if len(cases) != 200:
        raise ValueError(f"Generated {len(cases)} cases, expected 200")
    return cases
