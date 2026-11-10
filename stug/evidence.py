"""Three-valued evidence model (Sec. IV-C): Y (evidenced), N (evidenced false), ? (unknown)."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Tri(str, Enum):
    Y = "Y"
    N = "N"
    U = "?"

    def __bool__(self):  # guard against accidental truthiness
        raise TypeError("use explicit comparison with Tri.Y / Tri.N / Tri.U")

    @staticmethod
    def of(b: bool | None) -> "Tri":
        return Tri.U if b is None else (Tri.Y if b else Tri.N)

    def and_(self, other: "Tri") -> "Tri":
        if self is Tri.N or other is Tri.N:
            return Tri.N
        if self is Tri.Y and other is Tri.Y:
            return Tri.Y
        return Tri.U

    def or_(self, other: "Tri") -> "Tri":
        if self is Tri.Y or other is Tri.Y:
            return Tri.Y
        if self is Tri.N and other is Tri.N:
            return Tri.N
        return Tri.U


@dataclass
class Value:
    """An attribute value with provenance and confidence."""
    value: object
    prov: str = "static"        # static | dynamic
    conf: float = 1.0
    note: str = ""

    def to_dict(self):
        v = self.value.value if isinstance(self.value, Enum) else self.value
        return {"value": v, "prov": self.prov, "conf": round(self.conf, 2), "note": self.note}


# ---------------------------------------------------------------------------
# Strength lattices (Eqs. 7, 8)
# ---------------------------------------------------------------------------
INTEGRITY_LATTICE = {"none": 0, "additive": 1, "crc": 2, "md5": 3, "sha1": 3,
                     "sha2": 4, "sha3": 4, "blake2": 4}
AUTH_LATTICE = {"none": 0, "hardcoded_flag": 1, "hardcoded_model": 1,
                "mac_key_in_image": 2, "signature_key_in_image": 2,
                "mac": 3, "signature": 3}


def integrity_strength(alg: str) -> int:
    return INTEGRITY_LATTICE.get(alg, 0)


def auth_strength(kind: str) -> int:
    return AUTH_LATTICE.get(kind, 0)


@dataclass
class Report:
    """One property result (Algorithm 2 output entry)."""
    prop: str
    status: str                  # violated | satisfied | unresolved | n/a
    node: str = ""
    path: list = field(default_factory=list)
    detail: str = ""

    def to_dict(self):
        return {"prop": self.prop, "status": self.status, "node": self.node,
                "path": self.path, "detail": self.detail}
