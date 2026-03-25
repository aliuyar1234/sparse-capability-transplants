from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha1
from typing import Any

ALIAS_BANK_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class AliasBanks:
    banks: dict[str, dict[str, dict[str, list[str]]]]


def _stable_payload(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _rank_alias(canonical_key: str, alias: str) -> str:
    return sha1(f"{canonical_key}::{alias}".encode("utf-8")).hexdigest()


def _partition_aliases(canonical_key: str, aliases: list[str]) -> dict[str, list[str]]:
    ranked = sorted(
        _dedupe_preserve_order(aliases), key=lambda alias: _rank_alias(canonical_key, alias)
    )
    partition = {bank: [] for bank in ALIAS_BANK_NAMES}
    for index, alias in enumerate(ranked):
        bank = ALIAS_BANK_NAMES[index % len(ALIAS_BANK_NAMES)]
        partition[bank].append(alias)
    return partition


def _validate_domain_collisions(
    domain_banks: dict[str, dict[str, list[str]]], domain_name: str
) -> None:
    seen: dict[str, tuple[str, str]] = {}
    for bank_name, per_key_aliases in domain_banks.items():
        for canonical_key, aliases in per_key_aliases.items():
            for alias in aliases:
                previous = seen.get(alias)
                if previous is None:
                    seen[alias] = (bank_name, canonical_key)
                    continue
                previous_bank, previous_key = previous
                if previous_key != canonical_key:
                    raise ValueError(
                        "Alias collision detected in "
                        f"{domain_name!r}: alias {alias!r} assigned to "
                        f"{previous_key!r} ({previous_bank}) and {canonical_key!r} ({bank_name})"
                    )


def freeze_alias_banks(
    candidates_by_domain: dict[str, dict[str, list[str]]],
) -> AliasBanks:
    banks = {
        bank_name: {domain_name: {} for domain_name in candidates_by_domain}
        for bank_name in ALIAS_BANK_NAMES
    }

    for domain_name, candidates_by_key in candidates_by_domain.items():
        for canonical_key, aliases in candidates_by_key.items():
            partition = _partition_aliases(canonical_key, aliases)
            for bank_name, bank_aliases in partition.items():
                banks[bank_name][domain_name][canonical_key] = bank_aliases

        _validate_domain_collisions(
            {bank_name: banks[bank_name][domain_name] for bank_name in ALIAS_BANK_NAMES},
            domain_name=domain_name,
        )

    return AliasBanks(banks=banks)


def alias_banks_hash(alias_banks: AliasBanks) -> str:
    payload = _stable_payload(alias_banks.banks)
    return sha1(payload.encode("utf-8")).hexdigest()
