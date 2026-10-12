#!/usr/bin/env bash
#
# Downloads the four MNIST IDX files into ./data, verifies each one is a
# real IDX file (magic number + expected byte size), and decompresses
# them in place. Exits non-zero on the first failure so a CI pipeline or
# a chained command stops immediately.
#
# Usage:
#     bash download_mnist.sh            # writes to ./data
#     bash download_mnist.sh /tmp/mnist # writes to a specific directory
#

DATA_DIR="${1:-data}"
BASE_URL="https://storage.googleapis.com/cvdf-datasets/mnist"

# name:expected_size_bytes:expected_magic
#
# Sizes are exact; IDX headers plus the payload. Magic is the 4-byte
# big-endian value stored at offset 0: 2051 for image data, 2049 for
# label data. Checking both catches the two common failure modes:
# an HTML error page (wrong size, wrong magic) and a valid file that
# got truncated mid-download (right magic, wrong size).
FILES=(
    "train-images-idx3-ubyte:47040016:2051"
    "train-labels-idx1-ubyte:60008:2049"
    "t10k-images-idx3-ubyte:7840016:2051"
    "t10k-labels-idx1-ubyte:10008:2049"
)

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

for entry in "${FILES[@]}"; do
    IFS=':' read -r name expected_size expected_magic <<< "$entry"
    gz="${name}.gz"

    echo "==> $name"

    # -f  fail on HTTP >= 400 instead of saving the error body
    # -L  follow redirects (the mirror issues a 302 to a signed URL)
    # --retry handle transient network failures without user intervention
    # -o  explicit output name, so we always know what we downloaded
    curl -fL --retry 3 --retry-delay 2 -o "$gz" "${BASE_URL}/${gz}"

    # Decompress, replacing the .gz. gunzip checks the CRC32 and ISIZE
    # trailer, so a truncated or corrupted download fails here.
    gunzip -f "$gz"

    # Verify size. stat -f%z is the BSD/macOS form; stat -c%s is GNU.
    # Try both so the script works on Linux CLI as well as your favorito Mac.
    actual_size=$(stat -f%z "$name" 2>/dev/null || stat -c%s "$name")

    if [[ "$actual_size" != "$expected_size" ]]; then
        echo "    ERROR: $name has size $actual_size, expected $expected_size" >&2
        exit 1
    fi

    # Verify magic. Read the first 4 bytes and interpret as unsigned
    # big-endian integer. `xxd -p -l 4` prints them as 8 hex digits;
    # `printf '%d' 0x...` converts to decimal in bash without an extra
    # tool.
    magic_hex=$(xxd -p -l 4 "$name")
    magic_dec=$((16#$magic_hex))

    if [[ "$magic_dec" != "$expected_magic" ]]; then
        echo "    ERROR: $name has magic $magic_dec, expected $expected_magic" >&2
        echo "    First bytes: $(xxd -l 32 "$name")" >&2
        exit 1
    fi

    echo "    ok  size=$actual_size  magic=$magic_dec"
done

echo
echo "All four files verified in $(pwd)"
ls -lh