#include "mnist_loader.hpp"
#include <fstream>
#include <stdexcept>

/** Swaps the byte order of a 32-bit integer. MNIST stores all multi-byte
   integers in big-endian order (most significant byte first); x86 and
   ARM are little-endian (least significant byte first). If you don't trust me,
   just look up Stack Overflow. I did.
*/
static uint32_t reverse_int(uint32_t i) {
    return ((i & 0x000000FF) << 24) |
           ((i & 0x0000FF00) <<  8) |
           ((i & 0x00FF0000) >>  8) |
           ((i & 0xFF000000) >> 24);
}

void MNISTData::build_index() {
    // clear that before rebuilding it, otherwise you get duplicates if you call this twice,
    // by accident or on purpose
    by_label.clear();

    /* Pre-create all ten keys so iteration always yields exactly ten
       entries in order 0..9, even if some digit is absent from this split.
       reserve() avoids reallocation during the push_back loop below. */
    // MNIST has ten digit classes (0 through 9). Divide by 10 to estimate
    // the average number of samples per class for reserve(); this does not
    // divide or modify the labels themselves.
    const size_t per_class_estimate = labels.size() / 10;
    for (uint8_t d = 0; d < 10; ++d){
        by_label[d].reserve(per_class_estimate);
    }
    /* Single pass over the labels. operator[] on a map inserts a default
       value if the key is missing; here the keys already exist, so this
       is a pure lookup followed by an append. */
    for (size_t i = 0; i < labels.size(); ++i){
        by_label[labels[i]].push_back(i);
    }

}

MNISTData load_mnist(const std::string& data_dir,
                     const std::string& images_file,
                     const std::string& labels_file) {
    MNISTData data;

    /* Important-eh my liege, std::ios::binary disables newline translation. Without it, byte
       values that happen to match the platform's line-ending sequence
       would be silently rewritten during the read. */
    std::ifstream img_stream(data_dir + "/" + images_file, std::ios::binary);

    /* Fail immediately on a missing file rather than continuing with an
       empty stream, which would produce garbage downstream.
    */
    if (!img_stream) throw std::runtime_error("Cannot open " + images_file);

    uint32_t magic, num, rows, cols;

    /* std::istream::read takes a char*, because char is the byte-sized
       type with the loosest aliasing rules in C++. The header fields are
       uint32_t. There is no conversion defined between uint32_t* and
       char*, so static_cast would be a compile error. reinterpret_cast
       relabels the pointer's type without changing its value: the
       address the compiler writes into the call is identical, only the
       label differs. See the block comment at the top of this file.
    */
    img_stream.read(reinterpret_cast<char*>(&magic), 4);
    img_stream.read(reinterpret_cast<char*>(&num),   4);
    img_stream.read(reinterpret_cast<char*>(&rows),  4);
    img_stream.read(reinterpret_cast<char*>(&cols),  4);

    magic = reverse_int(magic);
    num   = reverse_int(num);
    rows  = reverse_int(rows);
    cols  = reverse_int(cols);

    /* 2051 is the IDX magic number for a 3-dimensional byte array
       (count, rows, cols).
       (maybe also SkyNet's year of destruction?)
       2049 is the 1-dimensional variant used for labels.
       These values come from the IDX file format specification, not
       from MNIST itself.
    */
    if (magic != 2051) throw std::runtime_error("Bad magic in " + images_file);
    if (rows != 28 || cols != 28)
        throw std::runtime_error("Unexpected image dimensions in " + images_file);

    /*
     * `num` is the number of images declared in the IDX header.
     * It is the same as the number of labels in the label file, and it is the total
     * sample count. For the standard MNIST training file,
     * num is 60,000; for the test file, it is 10,000.
     *
     * The first resize changes the outer vector, `data.images`. It creates
     * `num` Image objects so that data.images[i] is available for every
     * image index from 0 through num - 1. At this point the Image objects
     * themselves contain no pixel values yet.
     */
    data.images.resize(num);

    /*
     * `data.images` is a vector of Image objects, and Image is an alias for
     * `std::vector<uint8_t>`. Each inner vector stores one flattened image.
     *
     * The second resize runs once per image. `rows * cols` is the number of
     * pixels in that image. With the validated 28 by 28 MNIST dimensions,
     * each call allocates space for 28 * 28 = 784 bytes in data.images[i].
     * This is different from the first resize: the first creates the
     * collection of images, while this one creates the pixel buffer for one
     * particular image.
     *
     * The IDX file stores all image pixels in one continuous sequence, but
     * the nested vectors may use separate heap allocations. One read() per
     * image copies each 784-byte block into its corresponding inner vector.
     * The cast is the same pointer relabel as above: uint8_t* to char*, with
     * the address unchanged.
     */
    for (uint32_t i = 0; i < num; ++i) {
        data.images[i].resize(rows * cols);
        img_stream.read(reinterpret_cast<char*>(data.images[i].data()),
                        rows * cols);
    }

    std::ifstream lbl_stream(data_dir + "/" + labels_file, std::ios::binary);
    if (!lbl_stream) throw std::runtime_error("Cannot open " + labels_file);

    /*
     * Open the MNIST label file and read its two-field IDX header.
     *
     * The label file begins with:
     *
     *   1. A four-byte magic number, expected to be 2049.
     *   2. A four-byte count stating how many labels follow.
     *
     * `std::ifstream::read()` copies exactly four raw bytes into each
     * uint32_t. The IDX format stores integers in big-endian byte order,
     * while most Apple and Intel systems store integers in little-endian
     * order. `reverse_int()` therefore reverses the bytes after reading so
     * that `magic2` and `num2` have their normal numeric meanings.
     *
     * `reinterpret_cast<char*>` is required because the stream API accepts
     * a character buffer. It does not change the address or the bytes; it
     * only lets `read()` treat the uint32_t object's storage as raw input
     * memory. The values are validated below before the label payload is
     * consumed.
     */
    uint32_t magic2, num2;
    lbl_stream.read(reinterpret_cast<char*>(&magic2), 4);
    lbl_stream.read(reinterpret_cast<char*>(&num2),   4);
    magic2 = reverse_int(magic2);
    num2   = reverse_int(num2);

    if (magic2 != 2049) throw std::runtime_error("Bad magic in " + labels_file);

    /* The header declared num images and num2 labels. If they disagree,
       one of the two files is corrupt or the wrong file was supplied.
       Could be because curl fails to follow redirects or s3 (AWS-EAST *grins*)
       returned an HTML error page instead of the binary file.
       Catching this here avoids an out-of-bounds read later. */
    if (num2 != num)
        throw std::runtime_error("Image/label count mismatch in " + data_dir);

    /*
     * `lbl_stream` is the input file stream connected to the MNIST label
     * file. It remembers the file's current read position internally.
     * The two header reads above advanced that position past the eight-byte
     * IDX header, so the next read begins at the first label byte.
     *
     * `data.labels` is a single contiguous vector<uint8_t>, and the file
     * stores labels the same way, so one read() transfers the whole payload
     * in one operation. The reinterpret_cast is the same pointer relabel as
     * at every other binary read site: the byte buffer's address is
     * unchanged; only the type seen by the stream function changes.
     */
    data.labels.resize(num2);
    lbl_stream.read(reinterpret_cast<char*>(data.labels.data()), num2);

    return data;
}
/** reinterpret_cast.... whyyyyyy... why not static cast? Why not memcpy?
 * Why not a union, even if unholy? Why not a template function that takes
 * a pointer and a size and does the right thing?
 * Why not a library function that reads an integer from a stream in big-endian order?
 * Why not anything else?
 *
 * A pointer is an address. On this target, a 64-bit unsigned integer.
 * The type attached to it in source (uint32_t*, char*, Foo*) is a label
 * the compiler carries during parsing and semantic analysis and then
 * throws away. It never appears in generated code. At runtime, a
 * uint32_t* and a char* pointing at the same location are the same
 * eight bytes holding the same value.
 *
 * reinterpret_cast<T*>(p) changes which label the compiler associates
 * with that address. Compile a function with the cast and one
 * without, disassemble both, the machine code is identical. Op count: 0.
 * Life is a full circle. The cast is a lie, but the compiler believes it.
 *
 * Now look at reverse_int above. Big-endian to little-endian conversion
 * is real work. When this file casts &magic to char* and hands it
 * to read(), the four bytes coming off disk land in memory exactly
 * where they would have landed without the cast. Only the type that
 * gets checked against the read() signature has changed.
 *
 * The cast is needed because std::istream::read takes a char*. Whoever
 * wrote that signature had no way to know what type you wanted to fill,
 * so the parameter is char* viz the byte-sized type with the loosest
 * aliasing rules in C++ and the caller relabels. Every cast in this
 * file follows that pattern. Header fields are uint32_t, cast to char*,
 * four bytes land in the uint32_t. Image buffers are uint8_t*, cast to
 * char*, one byte per pixel lands in the right slot. Not one byte on
 * disk is transformed by any of it.
 *
 * What's given up is the compiler's checking. It can't verify that
 * destination and source agree on size, alignment, or lifetime.  The
 * image and label casts relabel uint8_t* as char*, the same byte under
 * a different name. Used between types that disagree on size or
 * alignment, the same cast would compile and would be undefined
 * behavior. The safety is in the usage, not the operator.
 */