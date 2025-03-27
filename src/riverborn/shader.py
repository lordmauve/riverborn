from importlib.resources import files
import re
import os
from typing import Dict, List, Tuple, Set, Optional, Union, Any

import moderngl
import moderngl_window as mglw


# Location information for a line of code
SourceLoc = Tuple[str, int, str]  # (filename, line_number, line_content)

shaders = files('riverborn') / "shaders"


def preprocess_shader(
    source: str,
    defines: Dict[str, str] = None,
    include_dirs: List[Any] = None,
    processed_includes: Set[str] = None,
    source_filename: str = "<string>",
) -> Union[str, Tuple[str, List[SourceLoc]]]:
    """Preprocess a shader source adding support for simple #include and #ifdef directives.

    Args:
        source: The shader source code to preprocess
        defines: Dictionary of define names and their values
        include_dirs: List of directories to search for include files
        processed_includes: Internal use only - set of already processed include files to avoid circularity
        source_filename: Name of the source file (for error reporting)

    Returns:
        Preprocessed shader source and source location mapping
    """
    defines = defines or {}
    include_dirs = include_dirs or [shaders]
    processed_includes = processed_includes or set()

    # Split source into lines and track source locations
    source_lines = source.splitlines()
    source_locs: List[SourceLoc] = []
    for i, line in enumerate(source_lines):
        source_locs.append((source_filename, i + 1, line))

    if not source_lines:
        return "", []

    # Process #include directives recursively
    i = 0
    result_lines = []
    result_locs = []

    while i < len(source_lines):
        line = source_lines[i]
        loc = source_locs[i]

        # Handle include directive
        include_match = re.match(r'^\s*#include\s+([<"].*?[>"])', line)
        if include_match:
            filename = include_match.group(1).strip('"').strip('<>')

            # Avoid circular includes
            if filename in processed_includes:
                result_lines.append(f"// Already included: {filename}")
                result_locs.append((loc[0], loc[1], f"// Already included: {filename}"))
                i += 1
                continue

            # Find and process the include file
            included_content = None
            included_file_path = None

            for dir_path in include_dirs:
                try:
                    if isinstance(dir_path, str):
                        filepath = os.path.join(dir_path, filename)
                        if os.path.exists(filepath):
                            with open(filepath, 'r') as f:
                                included_content = f.read()
                                included_file_path = filename
                                break
                    else:
                        # Handle Path-like objects from importlib.resources
                        include_path = dir_path / filename
                        if include_path.exists():
                            included_content = include_path.read_text()
                            included_file_path = filename
                            break
                except (FileNotFoundError, IsADirectoryError):
                    continue

            if included_content is None:
                raise FileNotFoundError(f"Could not find include file {filename}")

            # Add to processed includes to avoid circularity
            processed_includes.add(filename)

            # Process the included content recursively
            included_lines = included_content.splitlines()
            included_locs = [(included_file_path, j + 1, line) for j, line in enumerate(included_lines)]

            # Create a new set of processed includes to avoid modifying the original
            new_processed_includes = processed_includes.copy()

            # Process included content
            inner_proc_result = preprocess_shader(
                included_content,
                defines,
                include_dirs,
                new_processed_includes,
                included_file_path
            )

            if isinstance(inner_proc_result, tuple):
                inner_processed, inner_locs = inner_proc_result
            else:
                inner_processed = inner_proc_result
                inner_locs = included_locs

            # Add processed content to result
            inner_lines = inner_processed.splitlines()
            result_lines.extend(inner_lines)
            result_locs.extend(inner_locs)

            i += 1
            continue

        # Handle #ifdef
        ifdef_match = re.match(r'^\s*#ifdef\s+(\w+)\s*$', line)
        if ifdef_match:
            define_name = ifdef_match.group(1)
            is_defined = define_name in defines

            # Skip to matching #else or #endif
            i, lines_to_add, locs_to_add = process_conditional_block(
                source_lines, source_locs, i, defines, is_defined)

            result_lines.extend(lines_to_add)
            result_locs.extend(locs_to_add)
            continue

        # Handle #ifndef
        ifndef_match = re.match(r'^\s*#ifndef\s+(\w+)\s*$', line)
        if ifndef_match:
            define_name = ifndef_match.group(1)
            is_not_defined = define_name not in defines

            # Skip to matching #else or #endif
            i, lines_to_add, locs_to_add = process_conditional_block(
                source_lines, source_locs, i, defines, is_not_defined)

            result_lines.extend(lines_to_add)
            result_locs.extend(locs_to_add)
            continue

        # Handle #else without matching #ifdef/#ifndef
        else_match = re.match(r'^\s*#else\s*$', line)
        if else_match:
            raise SyntaxError("#else without matching #ifdef/#ifndef")

        # Handle #endif without matching #ifdef/#ifndef
        endif_match = re.match(r'^\s*#endif\s*$', line)
        if endif_match:
            raise SyntaxError("#endif without matching #ifdef/#ifndef")

        # Handle #define - keep the definition in the output
        define_match = re.match(r'^\s*#define\s+(\w+)(?:\s+(.*))?$', line)
        if define_match:
            define_name = define_match.group(1)
            define_value = define_match.group(2) or "1"
            defines[define_name] = define_value.strip()

            result_lines.append(line)
            result_locs.append(loc)
            i += 1
            continue

        # Add regular line to the result
        result_lines.append(line)
        result_locs.append(loc)
        i += 1

    processed_source = '\n'.join(result_lines)
    return processed_source, result_locs


def process_conditional_block(
    source_lines,
    source_locs,
    start_idx,
    defines,
    condition_result
) -> Tuple[int, List[str], List[SourceLoc]]:
    """Process a conditional block (#ifdef/#ifndef) and return the result.

    Args:
        source_lines: List of source lines
        source_locs: List of source locations
        start_idx: Index of the #ifdef/#ifndef line
        defines: Dictionary of preprocessor defines
        condition_result: Result of the condition (True/False)

    Returns:
        Tuple of (new_idx, lines_to_add, locs_to_add)
    """
    i = start_idx + 1  # Skip the opening directive
    nesting_level = 1
    lines_if_branch = []
    locs_if_branch = []
    lines_else_branch = []
    locs_else_branch = []

    in_else_branch = False

    while i < len(source_lines) and nesting_level > 0:
        line = source_lines[i]
        loc = source_locs[i]

        # Check for nested directives
        if re.match(r'^\s*#ifdef\s+\w+\s*$', line) or re.match(r'^\s*#ifndef\s+\w+\s*$', line):
            nesting_level += 1
            # For nested directives, we need to include the directive line in the output
            # for the chosen branch
            if not in_else_branch and condition_result:
                lines_if_branch.append(line)
                locs_if_branch.append(loc)
            elif in_else_branch and not condition_result:
                lines_else_branch.append(line)
                locs_else_branch.append(loc)
        elif re.match(r'^\s*#endif\s*$', line):
            nesting_level -= 1
            if nesting_level == 0:
                # End of this conditional block - don't include the #endif
                i += 1
                break
            # For nested directives, include the #endif in the output
            # for the chosen branch
            if not in_else_branch and condition_result:
                lines_if_branch.append(line)
                locs_if_branch.append(loc)
            elif in_else_branch and not condition_result:
                lines_else_branch.append(line)
                locs_else_branch.append(loc)
        elif nesting_level == 1 and re.match(r'^\s*#else\s*$', line):
            in_else_branch = True
        else:
            # Add line to appropriate branch
            if not in_else_branch:
                if condition_result:
                    lines_if_branch.append(line)
                    locs_if_branch.append(loc)
            else:
                if not condition_result:
                    lines_else_branch.append(line)
                    locs_else_branch.append(loc)

        i += 1

    if nesting_level > 0:
        raise SyntaxError("Unclosed #ifdef or #ifndef directive")

    # Return the appropriate branch based on condition result
    if condition_result:
        return i, lines_if_branch, locs_if_branch
    else:
        return i, lines_else_branch, locs_else_branch


def format_shader_with_line_info(source: str, source_locs: List[SourceLoc]) -> str:
    """Format shader source with line information for error reporting."""
    lines = source.splitlines()
    annotated_lines = []

    source_locs = [f'{filename}:{lineno}' for filename, lineno, _ in source_locs]
    max_loc_length = max(len(loc) for loc in source_locs)

    for lineno, (loc, line) in enumerate(zip(source_locs, lines), start=1):

        annotated_lines.append(f"\x1b[36m{lineno:<4}\x1b[0m \x1b[2m{loc:<{max_loc_length}}\x1b[0m {line}")

    return '\n'.join(annotated_lines)


def load_shader(name: str, *, vert: str | None = None, frag: str = None, defines: dict = None) -> moderngl.Program:
    """Load a shader from the shaders directory with preprocessing.

    Args:
        name: Base name of the shader
        vert: Optional vertex shader name (defaults to name)
        frag: Optional fragment shader name (defaults to name)
        defines: Optional dictionary of preprocessor defines

    Returns:
        Compiled shader program
    """
    vert = vert or name
    frag = frag or name

    ctx = mglw.ctx()

    try:
        # Load shader sources
        vert_path = f"{vert}.vert"
        frag_path = f"{frag}.frag"

        vert_source = (shaders / vert_path).read_text()
        frag_source = (shaders / frag_path).read_text()

        # Create a copy of defines to avoid modifying the original
        shader_defines = defines.copy() if defines else {}

        # Preprocess shaders with source mapping
        vert_result = preprocess_shader(
            vert_source,
            shader_defines,
            source_filename=str(vert_path)
        )

        frag_result = preprocess_shader(
            frag_source,
            shader_defines,
            source_filename=str(frag_path)
        )

        if isinstance(vert_result, tuple):
            vert_processed, vert_locs = vert_result
        else:
            vert_processed = vert_result
            vert_locs = []

        if isinstance(frag_result, tuple):
            frag_processed, frag_locs = frag_result
        else:
            frag_processed = frag_result
            frag_locs = []

        # Attempt to compile the shader program
        try:
            program = ctx.program(
                vertex_shader=vert_processed,
                fragment_shader=frag_processed,
            )
            return program
        except moderngl.Error as e:
            # Handle shader compilation errors with better diagnostics
            error_msg = str(e)

            # Determine which shader failed to compile
            if "vertex_shader" in error_msg:
                print("\nVertex shader compilation failed:")
                print(format_shader_with_line_info(vert_processed, vert_locs))
            elif "fragment_shader" in error_msg:
                print("\nFragment shader compilation failed:")
                print(format_shader_with_line_info(frag_processed, frag_locs))

            # Re-raise the error with the original message
            raise

    except Exception as e:
        print(f"Error loading shader {name}: {e}")
        # Print more detail for debugging
        if isinstance(e, SyntaxError) and "Unclosed #ifdef or #ifndef directive" in str(e):
            print(f"Check the shader files for unclosed directives:")
            print(f"  - {vert}.vert")
            print(f"  - {frag}.frag")
            if "shadow_common.glsl" in str(e):
                print("  - shaders/shadow_common.glsl")
        raise
