"""Operation-level intermediate representation shared by all frontends.

A frontend (shell, ELF, FirmUpdate adapter) turns update code into:
  * a list of Op records (refined call sites, Sec. IV-B)
  * a control-flow graph over op ids with ENTRY / EXIT_OK / EXIT_ERR sinks
Edge labels: 'ok' (command succeeded), 'fail' (command failed), 'any'.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import networkx as nx

from .evidence import Tri

ENTRY, EXIT_OK, EXIT_ERR = "ENTRY", "EXIT_OK", "EXIT_ERR"

# Op kinds (roles)
RECEIVE = "receive"          # wget/curl/tftp/recv/upload: new artifact enters
VERIFY = "verify"            # hash / CRC / signature / header-flag / model checks
VERSION_READ = "version_read"
VERSION_CHECK = "version_check"
COPY = "copy"                # identity-preserving (cp, mv, cat a > b)
TRANSFORM = "transform"      # content-preserving derivation (decompress, decrypt, untar)
MODIFY = "modify"            # in-place change of an artifact (sed -i, dd notrunc, echo >>)
WRITE = "write"              # write of an artifact to a storage target (flash or file)
ERASE = "erase"
BOOTSEL = "bootsel"          # boot slot / boot flag selection
RECOVERY = "recovery"        # boot counter, altbootcmd, rollback logic
REBOOT = "reboot"
MOUNT = "mount"
HASH = "hash_compute"        # hash computed but not (yet) compared
OTHER = "other"

INSTALL_LIKE = {WRITE}


@dataclass
class Op:
    id: str
    kind: str
    program: str
    line: int
    text: str
    src: str | None = None           # file read
    dst: str | None = None           # file / device written
    target: str | None = None        # partition name or device for flash utilities
    check_kind: str | None = None    # integrity | authenticity
    check_alg: str | None = None     # lattice key (crc, md5, sha2, signature, hardcoded_flag, ...)
    key_path: str | None = None
    covers: list = field(default_factory=list)   # files a verification covers
    var: str | None = None           # variable name (version read / compare)
    value: str | None = None         # value written (bootsel slot, env value)
    in_condition: bool = False
    on_fail_exit: Tri = Tri.U        # filled after CFG construction
    meta: dict = field(default_factory=dict)

    def to_dict(self):
        d = asdict(self)
        d["on_fail_exit"] = self.on_fail_exit.value
        return d


@dataclass
class UpdatePath:
    """Ops + CFG for one candidate update entry point."""
    name: str
    ops: dict[str, Op]
    cfg: nx.DiGraph
    program_files: list[str] = field(default_factory=list)
    set_e: bool = False

    def ops_of(self, *kinds) -> list[Op]:
        return [o for o in self.ops.values() if o.kind in kinds]

    def order(self) -> list[str]:
        """A deterministic topological-ish order (DFS preorder from ENTRY)."""
        seen, out = set(), []
        for n in nx.dfs_preorder_nodes(self.cfg, ENTRY):
            if n in self.ops and n not in seen:
                seen.add(n)
                out.append(n)
        return out


def compute_fail_handling(up: UpdatePath) -> None:
    """Set on_fail_exit for verify / version-check ops (result handling flag, Sec. IV-D).

    Y: the op has a failure edge and no write is reachable from it.
    N: the result is ignored ('any' edge) or a write is reachable after failure.
    ?: no information.
    """
    writes = {o.id for o in up.ops.values() if o.kind == WRITE}
    for op in up.ops.values():
        if op.kind not in (VERIFY, VERSION_CHECK):
            continue
        out = list(up.cfg.out_edges(op.id, data="label"))
        labels = {lab for _, _, lab in out}
        if "any" in labels and not up.set_e:
            op.on_fail_exit = Tri.N
            continue
        fail_succ = [v for _, v, lab in out if lab == "fail"]
        if not fail_succ:
            op.on_fail_exit = Tri.U
            continue
        reach = set()
        for s in fail_succ:
            reach |= {s} | nx.descendants(up.cfg, s)
        op.on_fail_exit = Tri.N if reach & writes else Tri.Y
