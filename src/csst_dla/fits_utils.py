from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct

import numpy as np


@dataclass(frozen=True)
class HduInfo:
    name: str
    header: dict
    data_offset: int
    data_size: int


def _parse_value(raw: str):
    raw = raw.strip()
    if raw.startswith("'"):
        return raw.strip().strip("'").strip()
    if raw in {"T", "F"}:
        return raw == "T"
    try:
        return int(raw)
    except ValueError:
        try:
            return float(raw)
        except ValueError:
            return raw


def read_hdu_infos(path: str | Path) -> list[HduInfo]:
    infos: list[HduInfo] = []
    with Path(path).open("rb") as f:
        while True:
            header_bytes = b""
            while True:
                block = f.read(2880)
                if not block:
                    return infos
                header_bytes += block
                if any(block[i : i + 3] == b"END" for i in range(0, 2880, 80)):
                    break

            header: dict = {}
            for i in range(0, len(header_bytes), 80):
                card = header_bytes[i : i + 80].decode("ascii", "replace")
                key = card[:8].strip()
                if key == "END":
                    break
                if key and card[8:10] == "= ":
                    header[key] = _parse_value(card[10:].split("/")[0])

            data_offset = f.tell()
            xtension = header.get("XTENSION", "PRIMARY")
            naxis = int(header.get("NAXIS", 0))
            if xtension == "BINTABLE":
                data_size = int(header["NAXIS1"]) * int(header["NAXIS2"]) + int(
                    header.get("PCOUNT", 0)
                )
            elif naxis:
                data_size = abs(int(header["BITPIX"])) // 8
                for axis in range(1, naxis + 1):
                    data_size *= int(header[f"NAXIS{axis}"])
            else:
                data_size = 0

            infos.append(
                HduInfo(
                    name=str(header.get("EXTNAME", xtension)).strip(),
                    header=header,
                    data_offset=data_offset,
                    data_size=data_size,
                )
            )
            f.seek(((data_size + 2879) // 2880) * 2880, 1)


def _hdu_by_name(path: str | Path, name: str) -> HduInfo:
    for info in read_hdu_infos(path):
        if info.name == name:
            return info
    raise KeyError(f"HDU {name!r} not found in {path}")


def read_image(path: str | Path, name: str) -> np.ndarray:
    info = _hdu_by_name(path, name)
    header = info.header
    bitpix = int(header["BITPIX"])
    if bitpix == -32:
        dtype = ">f4"
    elif bitpix == -64:
        dtype = ">f8"
    elif bitpix == 32:
        dtype = ">i4"
    elif bitpix == 16:
        dtype = ">i2"
    else:
        raise ValueError(f"Unsupported BITPIX={bitpix} for {name}")

    shape = tuple(int(header[f"NAXIS{i}"]) for i in range(int(header["NAXIS"]), 0, -1))
    with Path(path).open("rb") as f:
        f.seek(info.data_offset)
        data = np.frombuffer(f.read(info.data_size), dtype=dtype).copy()
    return data.reshape(shape)


def read_labels(path: str | Path) -> dict[str, np.ndarray]:
    info = _hdu_by_name(path, "LABELS")
    n_rows = int(info.header["NAXIS2"])
    row_size = int(info.header["NAXIS1"])
    fmt = ">fhhfffffff"
    fields = [
        "Z_QSO",
        "HAS_DLA",
        "N_DLA",
        "Z_DLA1",
        "LOGNHI1",
        "Z_DLA2",
        "LOGNHI2",
        "SNR_GU",
        "SNR_GV",
        "SNR_GI",
    ]
    with Path(path).open("rb") as f:
        f.seek(info.data_offset)
        raw = f.read(info.data_size)

    rows = [struct.unpack_from(fmt, raw, i * row_size) for i in range(n_rows)]
    arr = np.asarray(rows)
    out = {field: arr[:, i].copy() for i, field in enumerate(fields)}
    out["HAS_DLA"] = out["HAS_DLA"].astype(np.int16)
    out["N_DLA"] = out["N_DLA"].astype(np.int16)
    return out


def read_meta(path: str | Path) -> dict[str, np.ndarray]:
    info = _hdu_by_name(path, "META")
    n_rows = int(info.header["NAXIS2"])
    row_size = int(info.header["NAXIS1"])
    with Path(path).open("rb") as f:
        f.seek(info.data_offset)
        raw = f.read(info.data_size)
    rows = [struct.unpack_from(">qf", raw, i * row_size) for i in range(n_rows)]
    arr = np.asarray(rows)
    return {"TARGETID": arr[:, 0].astype(np.int64), "Z_QSO": arr[:, 1].astype(np.float32)}


def _card(key: str, value=None, comment: str | None = None) -> str:
    if value is None:
        text = key
    else:
        if isinstance(value, str):
            rendered = f"'{value:<8}'" if len(value) <= 8 else f"'{value}'"
        elif isinstance(value, bool):
            rendered = "T" if value else "F"
        else:
            rendered = str(value)
        text = f"{key:<8}= {rendered:>20}"
        if comment:
            text += f" / {comment}"
    return text[:80].ljust(80)


def _header(cards: list[str]) -> bytes:
    text = "".join(cards + ["END".ljust(80)]).encode("ascii")
    return text + b" " * (((len(text) + 2879) // 2880) * 2880 - len(text))


def write_catalog_fits(
    path: str | Path,
    targetid: np.ndarray,
    z_qso: np.ndarray,
    z_dla: np.ndarray,
    log_nhi: np.ndarray,
    confidence: np.ndarray,
) -> None:
    targetid = np.asarray(targetid, dtype=">i8")
    z_qso = np.asarray(z_qso, dtype=">f4")
    z_dla = np.asarray(z_dla, dtype=">f4")
    log_nhi = np.asarray(log_nhi, dtype=">f4")
    confidence = np.asarray(confidence, dtype=">f4")
    n_rows = len(targetid)

    primary = _header(
        [
            _card("SIMPLE", True),
            _card("BITPIX", 8),
            _card("NAXIS", 0),
            _card("EXTEND", True),
        ]
    )

    row_size = 24
    table_header = _header(
        [
            _card("XTENSION", "BINTABLE"),
            _card("BITPIX", 8),
            _card("NAXIS", 2),
            _card("NAXIS1", row_size),
            _card("NAXIS2", n_rows),
            _card("PCOUNT", 0),
            _card("GCOUNT", 1),
            _card("TFIELDS", 5),
            _card("TTYPE1", "TARGETID"),
            _card("TFORM1", "K"),
            _card("TTYPE2", "Z_QSO"),
            _card("TFORM2", "E"),
            _card("TTYPE3", "Z_DLA"),
            _card("TFORM3", "E"),
            _card("TTYPE4", "LOG_NHI"),
            _card("TFORM4", "E"),
            _card("TTYPE5", "CONFIDENCE"),
            _card("TFORM5", "E"),
            _card("EXTNAME", "CATALOG"),
        ]
    )

    data = bytearray(n_rows * row_size)
    for i in range(n_rows):
        struct.pack_into(
            ">qffff",
            data,
            i * row_size,
            int(targetid[i]),
            float(z_qso[i]),
            float(z_dla[i]),
            float(log_nhi[i]),
            float(confidence[i]),
        )
    padding = b"\0" * (((len(data) + 2879) // 2880) * 2880 - len(data))
    Path(path).write_bytes(primary + table_header + bytes(data) + padding)

