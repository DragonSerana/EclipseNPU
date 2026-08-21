#!/usr/bin/env python3
"""Static hazard checker for Eclipse .easm traces.

Reads an .easm instruction stream and reports RAW/WAR/WAW hazards that are
not separated by a SYNC fence.
"""

import sys

DTYPE_SIZE = 2


def die(message):
    print(f"error: {message}", file=sys.stderr)
    sys.exit(1)


def parse_hex(text):
    return int(text, 0)


def tile_ranges(base, rows, cols, stride):
    """Return the list of half-open byte intervals touched by a 2D tile."""
    ranges = []
    for r in range(rows):
        start = base + r * stride
        ranges.append((start, start + cols * DTYPE_SIZE))
    return ranges


def ranges_overlap(a_ranges, b_ranges):
    for a0, a1 in a_ranges:
        for b0, b1 in b_ranges:
            if a0 < b1 and b0 < a1:
                return True
    return False


def parse_easm(path):
    """Return a list of (opcode, fields) tuples, one per non-empty line."""
    instructions = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            parts = line.split()
            op = parts[0]
            fields = {}
            for part in parts[1:]:
                if "=" in part:
                    key, value = part.split("=", 1)
                    fields[key] = value
            instructions.append((op, fields))
    return instructions


def parse_instruction(op, fields):
    """Return (reads, writes) for one instruction.

    Each element is a list of (name, ranges) where ranges is a list of
    half-open byte intervals.
    """
    reads = []
    writes = []

    if op in ("DMA_LOAD", "DMA_STORE"):
        sram = parse_hex(fields["sram"])
        ddr = parse_hex(fields["ddr"])
        rows = int(fields["rows"])
        cols = int(fields["cols"])
        src_stride = int(fields["srcStride"])
        dst_stride = int(fields["dstStride"])

        if op == "DMA_LOAD":
            reads.append(("DDR", tile_ranges(ddr, rows, cols, src_stride)))
            writes.append(("SRAM", tile_ranges(sram, rows, cols, dst_stride)))
        else:
            reads.append(("SRAM", tile_ranges(sram, rows, cols, src_stride)))
            writes.append(("DDR", tile_ranges(ddr, rows, cols, dst_stride)))

    elif op == "MATMUL":
        dst = parse_hex(fields["dst"])
        lhs = parse_hex(fields["lhs"])
        rhs = parse_hex(fields["rhs"])
        m = int(fields["M"])
        n = int(fields["N"])
        k = int(fields["K"])
        acc = int(fields["acc"])

        reads.append(("lhs", [(lhs, lhs + m * k * DTYPE_SIZE)]))
        reads.append(("rhs", [(rhs, rhs + k * n * DTYPE_SIZE)]))
        if acc:
            reads.append(("dst", [(dst, dst + m * n * DTYPE_SIZE)]))
        writes.append(("dst", [(dst, dst + m * n * DTYPE_SIZE)]))

    elif op == "ELEMENTWISE_ADD":
        dst = parse_hex(fields["dst"])
        lhs = parse_hex(fields["lhs"])
        rhs = parse_hex(fields["rhs"])
        n = int(fields["n"])

        reads.append(("lhs", [(lhs, lhs + n * DTYPE_SIZE)]))
        reads.append(("rhs", [(rhs, rhs + n * DTYPE_SIZE)]))
        writes.append(("dst", [(dst, dst + n * DTYPE_SIZE)]))

    elif op == "ACT":
        dst = parse_hex(fields["dst"])
        src = parse_hex(fields["src"])
        n = int(fields["n"])

        reads.append(("src", [(src, src + n * DTYPE_SIZE)]))
        writes.append(("dst", [(dst, dst + n * DTYPE_SIZE)]))

    elif op == "SYNC":
        pass

    else:
        die(f"unknown opcode: {op}")

    return reads, writes


def main():
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <trace.easm>", file=sys.stderr)
        return 2

    instructions = parse_easm(sys.argv[1])
    active = []
    violations = []

    for index, (op, fields) in enumerate(instructions, start=1):
        if op == "SYNC":
            active.clear()
            continue

        reads, writes = parse_instruction(op, fields)

        # RAW: current read conflicts with an earlier write.
        for read_name, read_ranges in reads:
            for prev_index, prev_op, prev_kind, prev_name, prev_ranges in active:
                if prev_kind == "write" and ranges_overlap(read_ranges, prev_ranges):
                    violations.append(
                        (index, op, "RAW", f"read {read_name}", prev_index,
                         prev_op, f"write {prev_name}"))

        # WAR/WAW: current write conflicts with earlier read/write.
        for write_name, write_ranges in writes:
            for prev_index, prev_op, prev_kind, prev_name, prev_ranges in active:
                if prev_kind == "read" and ranges_overlap(write_ranges, prev_ranges):
                    violations.append(
                        (index, op, "WAR", f"write {write_name}", prev_index,
                         prev_op, f"read {prev_name}"))
                elif prev_kind == "write" and ranges_overlap(write_ranges, prev_ranges):
                    violations.append(
                        (index, op, "WAW", f"write {write_name}", prev_index,
                         prev_op, f"write {prev_name}"))

        for read_name, read_ranges in reads:
            active.append((index, op, "read", read_name, read_ranges))
        for write_name, write_ranges in writes:
            active.append((index, op, "write", write_name, write_ranges))

    if not violations:
        print("no hazards")
        return 0

    print(f"{len(violations)} hazard(s) found:")
    for cur_index, cur_op, hazard_type, cur_desc, prev_index, prev_op, prev_desc in violations:
        print(f"  instr {cur_index} ({cur_op}) {cur_desc} conflicts with "
              f"instr {prev_index} ({prev_op}) {prev_desc}: {hazard_type}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
