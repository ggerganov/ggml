#!/usr/bin/env python3
# gguf-addfile.py srcfile dstfile addfiles ...

from __future__ import annotations

import logging
import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Necessary to load the local gguf package
if "NO_LOCAL_GGUF" not in os.environ and (Path(__file__).parent.parent.parent / 'gguf-py').exists():
    #print("add path", str(Path(__file__).parent.parent))
    sys.path.insert(0, str(Path(__file__).parent.parent))

from gguf import GGUFReader, GGUFWriter, ReaderField, GGUFEndian, GGUFValueType, Keys, NamedObject  # noqa: E402

logger = logging.getLogger("gguf-addfile")


def get_file_host_endian(reader: GGUFReader) -> tuple[str, str]:
    host_endian = 'LITTLE' if np.uint32(1) == np.uint32(1).newbyteorder("<") else 'BIG'
    if reader.byte_order == 'S':
        file_endian = 'BIG' if host_endian == 'LITTLE' else 'LITTLE'
    else:
        file_endian = host_endian
    return (host_endian, file_endian)


# For more information about what field.parts and field.data represent,
# please see the comments in the modify_gguf.py example.
def dump_metadata(reader: GGUFReader, args: argparse.Namespace) -> None:
    host_endian, file_endian = get_file_host_endian(reader)
    print(f'* File is {file_endian} endian, script is running on a {host_endian} endian host.')
    print(f'\n* Dumping {len(reader.fields)} key/value pair(s)')
    for n, field in enumerate(reader.fields.values(), 1):
        if not field.types:
            pretty_type = 'N/A'
        elif field.types[0] == GGUFValueType.ARRAY:
            nest_count = len(field.types) - 1
            pretty_type = '[' * nest_count + str(field.types[-1].name) + ']' * nest_count
        else:
            pretty_type = str(field.types[-1].name)
        print(f'  {n:5}: {pretty_type:11} | {len(field.data):8} | {field.name}', end = '')
        if len(field.types) == 1:
            curr_type = field.types[0]
            if curr_type == GGUFValueType.STRING:
                print(' = {0}'.format(repr(str(bytes(field.parts[-1]), encoding='utf8')[:60])), end = '')
            elif curr_type == GGUFValueType.NAMEDOBJECT:
                print(' = {0}'.format(repr(str(bytes(field.parts[4]), encoding='utf8')[:60])), end = '')
                print(', {0}'.format(int(field.parts[5]))[:20], end = '')
            elif field.types[0] in reader.gguf_scalar_to_np:
                print(' = {0}'.format(field.parts[-1][0]), end = '')
        print()
    if args.no_tensors:
        return
    print(f'\n* Dumping {len(reader.tensors)} tensor(s)')
    for n, tensor in enumerate(reader.tensors, 1):
        prettydims = ', '.join('{0:5}'.format(d) for d in list(tensor.shape) + [1] * (4 - len(tensor.shape)))
        print(f'  {n:5}: {tensor.n_elements:10} | {prettydims} | {tensor.tensor_type.name:7} | {tensor.name}')


def dump_metadata_json(reader: GGUFReader, args: argparse.Namespace) -> None:
    import json
    host_endian, file_endian = get_file_host_endian(reader)
    metadata: dict[str, Any] = {}
    tensors: dict[str, Any] = {}
    result = {
        "filename": args.input,
        "endian": file_endian,
        "metadata": metadata,
        "tensors": tensors,
    }
    for idx, field in enumerate(reader.fields.values()):
        curr: dict[str, Any] = {
            "index": idx,
            "type": field.types[0].name if field.types else 'UNKNOWN',
            "offset": field.offset,
        }
        metadata[field.name] = curr
        if field.types[:1] == [GGUFValueType.ARRAY]:
            curr["array_types"] = [t.name for t in field.types][1:]
            if not args.json_array:
                continue
            itype = field.types[-1]
            if itype == GGUFValueType.STRING:
                curr["value"] = [str(bytes(field.parts[idx]), encoding="utf-8") for idx in field.data]
            elif itype == GGUFValueType.NAMEDOBJECT:
                curr["value"] = [str(bytes(field.parts[idx]), encoding="utf-8") for idx in field.data]
            else:
                curr["value"] = [pv for idx in field.data for pv in field.parts[idx].tolist()]
        elif field.types[0] == GGUFValueType.STRING:
            curr["value"] = str(bytes(field.parts[-1]), encoding="utf-8")
        elif field.types[0] == GGUFValueType.NAMEDOBJECT:
            curr["value"] = str(bytes(field.parts[4]), encoding="utf-8")
            curr["value"] = int(field.parts[5])
        else:
            curr["value"] = field.parts[-1].tolist()[0]
    if not args.no_tensors:
        for idx, tensor in enumerate(reader.tensors):
            tensors[tensor.name] = {
                "index": idx,
                "shape": tensor.shape.tolist(),
                "type": tensor.tensor_type.name,
                "offset": tensor.field.offset,
            }
    json.dump(result, sys.stdout)


def get_byteorder(reader: GGUFReader) -> GGUFEndian:
    if np.uint32(1) == np.uint32(1).newbyteorder("<"):
        # Host is little endian
        host_endian = GGUFEndian.LITTLE
        swapped_endian = GGUFEndian.BIG
    else:
        # Sorry PDP or other weird systems that don't use BE or LE.
        host_endian = GGUFEndian.BIG
        swapped_endian = GGUFEndian.LITTLE

    if reader.byte_order == "S":
        return swapped_endian
    else:
        return host_endian


def decode_field(field: ReaderField) -> Any:
    if field and field.types:
        main_type = field.types[0]

        if main_type == GGUFValueType.ARRAY:
            sub_type = field.types[-1]

            if sub_type == GGUFValueType.STRING:
                return [str(bytes(field.parts[idx]), encoding='utf8') for idx in field.data]
            elif sub_type == GGUFValueType.NAMEDOBJECT:
                return [str(bytes(field.parts[idx]), encoding='utf8') for idx in field.data]
            else:
                return [pv for idx in field.data for pv in field.parts[idx].tolist()]
        if main_type == GGUFValueType.STRING:
            return str(bytes(field.parts[-1]), encoding='utf8')
        elif main_type == GGUFValueType.NAMEDOBJECT:
            return str(bytes(field.parts[4]), encoding='utf8')
        else:
            return field.parts[-1][0]

    return None


def get_field_data(reader: GGUFReader, key: str) -> Any:
    field = reader.get_field(key)

    return decode_field(field)


def copy_with_new_metadata(reader: gguf.GGUFReader, writer: gguf.GGUFWriter, new_metadata: Mapping[str, str], array: NamedObject[Any] | None = None) -> None:
    for field in reader.fields.values():
        # Suppress virtual fields and fields written by GGUFWriter
        if field.name == Keys.General.ARCHITECTURE or field.name.startswith('GGUF.'):
            logger.debug(f'Suppressing {field.name}')
            continue

        # Skip old chat templates if we have new ones
        if field.name.startswith(Keys.Tokenizer.CHAT_TEMPLATE) and Keys.Tokenizer.CHAT_TEMPLATE in new_metadata:
            logger.debug(f'Skipping {field.name}')
            continue

        old_val = decode_field(field)
        val = new_metadata.get(field.name, old_val)

        if field.name in new_metadata:
            logger.debug(f'Modifying {field.name}: "{old_val}" -> "{val}"')
            del new_metadata[field.name]
        elif val is not None:
            logger.debug(f'Copying {field.name}')

        if val is not None:
            writer.add_key(field.name)
            writer.add_val(val, field.types[0])

    if Keys.Tokenizer.CHAT_TEMPLATE in new_metadata:
        logger.debug('Adding chat template(s)')
        writer.add_chat_template(new_metadata[Keys.Tokenizer.CHAT_TEMPLATE])
        del new_metadata[Keys.Tokenizer.CHAT_TEMPLATE]

    if array is None:
        for key, name in new_metadata.items():
            logger.debug(f'Adding {key}: {name}')
            # named object
            with open(name, "rb") as f:
                val = f.read()
                writer.add_namedobject(key, val, name)
    else:
        for key, name in new_metadata.items():
            logger.debug(f'Adding array {key}: {name}')
            # named object
            writer.add_namedobject(key, 'val', name, array=array)
    
    for tensor in reader.tensors:
        # Dimensions are written in reverse order, so flip them first
        shape = np.flipud(tensor.shape)
        writer.add_tensor_info(tensor.name, shape, tensor.data.dtype, tensor.data.nbytes, tensor.tensor_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    for tensor in reader.tensors:
        writer.write_tensor_data(tensor.data)

    writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Add files to GGUF file metadata")
    parser.add_argument("input",        type=str,            help="GGUF format model input filename")
    parser.add_argument("output",       type=str,            help="GGUF format model output filename")
    parser.add_argument("addfiles",     type=str, nargs='+', help="add filenames ...")
    parser.add_argument("--array",      action="store_true", help="add files to namedobject array")
    parser.add_argument("--no-tensors", action="store_true", help="Don't dump tensor metadata")
    parser.add_argument("--json",       action="store_true", help="Produce JSON output")
    parser.add_argument("--json-array", action="store_true", help="Include full array values in JSON output (long)")
    parser.add_argument("--verbose",    action="store_true", help="Increase output verbosity")
    args = parser.parse_args(None if len(sys.argv) > 1 else ["--help"])
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    logger.info(f'* Loading: {args.input}')
    reader = GGUFReader(args.input, 'r')
    arch = get_field_data(reader, Keys.General.ARCHITECTURE)
    endianess = get_byteorder(reader)

    logger.info(f'* Writing: {args.output}')
    writer = GGUFWriter(args.output, arch=arch, endianess=endianess)

    alignment = get_field_data(reader, Keys.General.ALIGNMENT)
    if alignment is not None:
        logger.debug(f'Setting custom alignment: {alignment}')
        writer.data_alignment = alignment

    logger.info(f'* Adding: {args.addfiles}')
    new_metadata = {}
    count = 0
    if args.array is False:
        for path in args.addfiles:
            count += 1
            key = Keys.General.NAMEDOBJECT + Keys.General.CONNECT + str(count)
            new_metadata[key] = path
            logger.info(f'* Adding: {key} = {path}')
        copy_with_new_metadata(reader, writer, new_metadata)
    else:
        key = Keys.General.NAMEDOBJECT
        # array is dummy
        new_metadata[key] = 'array'
        files = []
        for path in args.addfiles:
            with open(path, "rb") as f:
                val = f.read()
                #print(f'files[{count}] = {path}')
                files.append(NamedObject(path, val))
            logger.info(f'* Adding: {key}[{count}] = {path}')
            count += 1
        copy_with_new_metadata(reader, writer, new_metadata, array=files)

    if args.json:
        dump_metadata_json(reader, args)
    else:
        dump_metadata(reader, args)

    logger.info(f'* Reading: {args.output}')
    reader = GGUFReader(args.output, 'r')
    dump_metadata(reader, args)


if __name__ == '__main__':
    main()
