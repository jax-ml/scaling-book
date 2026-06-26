#!/usr/bin/env python3
"""
Convert all .md files in the scaling-book to a single combined markdown file
with standardized formatting.
"""

import re
from pathlib import Path

# Order of sections based on the book structure
SECTION_ORDER = [
    ("index.md", "Introduction", 0),
    ("roofline.md", "All About Rooflines", 1),
    ("tpus.md", "How to Think About TPUs", 2),
    ("sharding.md", "Sharded Matrices and How to Multiply Them", 3),
    ("transformers.md", "All the Transformer Math You Need to Know", 4),
    ("training.md", "How to Parallelize a Transformer for Training", 5),
    ("applied-training.md", "Training LLaMA 3 on TPUs", 6),
    ("inference.md", "All About Transformer Inference", 7),
    ("applied-inference.md", "Serving LLaMA 3 on TPUs", 8),
    ("profiling.md", "How to Profile TPU Code", 9),
    ("jax-stuff.md", "Programming TPUs in JAX", 10),
    ("conclusion.md", "Conclusions and Further Reading", 11),
    ("gpus.md", "How to Think About GPUs", 12),
]

# Mapping from file names to section anchors
FILE_TO_ANCHOR = {
    "index": "introduction",
    "roofline": "all-about-rooflines",
    "tpus": "how-to-think-about-tpus",
    "sharding": "sharded-matrices-and-how-to-multiply-them",
    "transformers": "all-the-transformer-math-you-need-to-know",
    "training": "how-to-parallelize-a-transformer-for-training",
    "applied-training": "training-llama-3-on-tpus",
    "inference": "all-about-transformer-inference",
    "applied-inference": "serving-llama-3-on-tpus",
    "profiling": "how-to-profile-tpu-code",
    "jax-stuff": "programming-tpus-in-jax",
    "conclusion": "conclusions-and-further-reading",
    "gpus": "how-to-think-about-gpus",
}


def strip_yaml_frontmatter(content: str) -> str:
    """Remove YAML frontmatter from markdown content."""
    if content.startswith("---"):
        # Find the closing ---
        second_delim = content.find("---", 3)
        if second_delim != -1:
            return content[second_delim + 3:].lstrip()
    return content


def convert_figure_includes(content: str) -> str:
    """Convert {% include figure.liquid ... %} to standard markdown images."""
    # Pattern for figure includes with various attributes
    # Use .*? (non-greedy) with DOTALL flag to handle multi-line captions
    pattern = r'\{%\s*include\s+figure\.liquid\s+(.*?)\s*%\}'

    def replace_figure(match):
        attrs = match.group(1)
        # Extract path
        path_match = re.search(r'path="([^"]+)"', attrs)
        if not path_match:
            return match.group(0)  # Return original if no path
        path = path_match.group(1)

        # Extract caption if present - handle captions that may span multiple lines
        caption_match = re.search(r'caption="(.*?)"(?:\s*%\}|\s+\w+)', attrs, re.DOTALL)
        if not caption_match:
            # Try simpler pattern
            caption_match = re.search(r'caption="([^"]*)"', attrs, re.DOTALL)
        caption = caption_match.group(1) if caption_match else ""

        # Clean up caption - remove HTML tags and normalize whitespace
        caption = re.sub(r'<[^>]+>', '', caption)
        caption = re.sub(r'\s+', ' ', caption)
        caption = caption.strip()

        # Generate markdown image
        if caption:
            return f'![{caption}]({path})'
        else:
            return f'![]({path})'

    return re.sub(pattern, replace_figure, content, flags=re.DOTALL)


def convert_internal_links(content: str) -> str:
    """Convert internal page links to anchor links."""
    # Pattern for links like [text](roofline) or [text](../gpus)
    # But not external links (http/https)

    def replace_link(match):
        text = match.group(1)
        target = match.group(2)

        # Skip external links
        if target.startswith(('http://', 'https://', 'mailto:')):
            return match.group(0)

        # Skip anchor-only links
        if target.startswith('#'):
            return match.group(0)

        # Skip image links and asset paths
        if 'assets/' in target or target.endswith(('.png', '.gif', '.jpg', '.jpeg', '.svg')):
            return match.group(0)

        # Extract the file name without ../ or ./ prefix
        clean_target = target
        while clean_target.startswith('../') or clean_target.startswith('./'):
            clean_target = clean_target[3:] if clean_target.startswith('../') else clean_target[2:]

        # Check if there's an anchor in the target
        anchor_part = ""
        if '#' in clean_target:
            clean_target, anchor_part = clean_target.split('#', 1)

        # Look up the section anchor
        if clean_target in FILE_TO_ANCHOR:
            new_anchor = FILE_TO_ANCHOR[clean_target]
            if anchor_part:
                return f'[{text}](#{anchor_part})'
            else:
                return f'[{text}](#{new_anchor})'

        # If not found, just return the original
        return match.group(0)

    # Match markdown links: [text](target)
    pattern = r'\[([^\]]+)\]\(([^)]+)\)'
    return re.sub(pattern, replace_link, content)


def convert_details_blocks(content: str) -> str:
    """Convert {% details %} blocks to standard markdown/HTML details."""
    # Convert opening tag
    content = re.sub(
        r'\{%\s*details\s+([^%]+)\s*%\}',
        r'<details>\n<summary>\1</summary>\n',
        content
    )
    # Convert closing tag
    content = re.sub(r'\{%\s*enddetails\s*%\}', '</details>', content)
    return content


def convert_footnotes(content: str, section_prefix: str) -> tuple[str, list[str]]:
    """Convert <d-footnote> tags to pandoc-style footnotes [^prefix-N].

    Args:
        content: The markdown content to process
        section_prefix: A unique prefix for this section (e.g., "ch1") to avoid
                        duplicate footnote IDs across sections

    Returns:
        Tuple of (converted content, list of (footnote_id, footnote_text))
    """
    footnotes: list[tuple[str, str]] = []

    def replace_footnote(match):
        text = match.group(1)
        # Clean up any HTML in the footnote
        text = re.sub(r'<a[^>]*>([^<]*)</a>', r'\1', text)
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        fn_id = f"{section_prefix}-{len(footnotes) + 1}"
        footnotes.append((fn_id, text))
        return f'[^{fn_id}]'

    pattern = r'<d-footnote>([^<]*(?:<(?!/?d-footnote)[^<]*)*)</d-footnote>'
    converted = re.sub(pattern, replace_footnote, content, flags=re.DOTALL)
    return converted, footnotes


def remove_citations(content: str) -> str:
    """Remove <d-cite> tags."""
    return re.sub(r'<d-cite\s+key="[^"]*"\s*></d-cite>', '', content)


def strip_unsupported_latex(content: str, filename: str) -> str:
    """Strip unsupported LaTeX commands, emit warnings."""

    # 1. Fix double-backslash braces: \\{ and \\} -> \{ and \}
    double_brace_matches = re.findall(r'\\\\[\{\}]', content)
    if double_brace_matches:
        print(f"  Warning: Found {len(double_brace_matches)} instances of \\\\{{ or \\\\}} in {filename}, converting to \\{{ \\}}")
        content = re.sub(r'\\\\(\{)', r'\\{', content)
        content = re.sub(r'\\\\(\})', r'\\}', content)

    # 2. Strip \textcolor{color}{text} -> text
    textcolor_matches = re.findall(r'\\textcolor\{[^}]+\}\{[^}]+\}', content)
    if textcolor_matches:
        print(f"  Warning: Found {len(textcolor_matches)} instances of \\textcolor in {filename}, stripping color:")
        for m in textcolor_matches[:3]:  # Show first 3
            print(f"    {m}")
        if len(textcolor_matches) > 3:
            print(f"    ... and {len(textcolor_matches) - 3} more")
        content = re.sub(r'\\textcolor\{[^}]+\}\{([^}]+)\}', r'\1', content)

    # 3. Strip \textnormal{text} -> text
    textnormal_matches = re.findall(r'\\textnormal\{[^}]+\}', content)
    if textnormal_matches:
        print(f"  Warning: Found {len(textnormal_matches)} instances of \\textnormal in {filename}, stripping:")
        for m in textnormal_matches[:3]:
            print(f"    {m}")
        content = re.sub(r'\\textnormal\{([^}]+)\}', r'\1', content)

    # 4. Strip size commands: \small, \tiny, etc.
    size_cmds = ['small', 'tiny', 'footnotesize', 'scriptsize', 'large', 'Large', 'huge', 'Huge']
    for cmd in size_cmds:
        pattern = r'\\' + cmd + r'(?:\s|$|\{)'
        matches = re.findall(pattern, content)
        if matches:
            print(f"  Warning: Found {len(matches)} instances of \\{cmd} in {filename}, stripping")
        content = re.sub(r'\\' + cmd + r'\s*', '', content)

    return content


def convert_inline_math(content: str) -> str:
    """Convert inline $$ math to single $ (leave block math alone)."""
    lines = content.split('\n')
    result = []

    for line in lines:
        # Skip lines that are just $$ (block math delimiters)
        stripped = line.strip()
        if stripped == '$$':
            result.append(line)
            continue

        # For other lines, convert inline $$...$$ to $...$
        # Match $$ that has content on the same line (inline math)
        # Use a pattern that finds $$...$$ where there's text before or after on the same line
        if '$$' in line and stripped != '$$':
            # Replace $$...$$ with $...$ for inline math
            line = re.sub(r'\$\$([^\$]+?)\$\$', r'$\1$', line)

        result.append(line)

    return '\n'.join(result)


def clean_html_elements(content: str) -> str:
    """Clean up custom HTML elements."""
    # Convert <p markdown=1 class="takeaway">**...**</p> to just the content
    content = re.sub(r'<p\s+markdown=1[^>]*>', '', content)
    content = re.sub(r'</p>', '', content)

    # Convert <h3 markdown=1 class="next-section">...</h3> to regular h3
    content = re.sub(r'<h3\s+markdown=1[^>]*>', '### ', content)
    content = re.sub(r'</h3>', '', content)

    # Remove <sup>*</sup> markers
    content = re.sub(r'<sup>\*</sup>', '*', content)

    # Clean up <b markdown=1 style=...> tags
    content = re.sub(r'<b\s+markdown=1[^>]*>', '**', content)
    content = re.sub(r'</b>', '**', content)

    # Clean up <span> tags with styling
    content = re.sub(r'<span[^>]*>', '', content)
    content = re.sub(r'</span>', '', content)

    return content


def slugify(text: str) -> str:
    """Convert text to a URL-friendly slug."""
    text = text.lower()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[-\s]+', '-', text)
    return text.strip('-')


def process_file(filepath: Path, section_title: str, section_num: int) -> str:
    """Process a single markdown file."""
    content = filepath.read_text(encoding='utf-8')

    # Strip frontmatter
    content = strip_yaml_frontmatter(content)

    # Create a unique prefix for footnotes based on section number
    section_prefix = f"ch{section_num}"

    # Apply conversions - order matters!
    # First convert figures before cleaning HTML (figures have HTML in captions)
    content = convert_figure_includes(content)
    content = convert_details_blocks(content)
    content, footnotes = convert_footnotes(content, section_prefix)
    content = remove_citations(content)
    # Clean HTML last so figure captions are already processed
    content = clean_html_elements(content)
    content = convert_internal_links(content)
    # Convert inline $$ to $ (keep block $$ on own lines)
    content = convert_inline_math(content)
    # Strip unsupported LaTeX commands
    content = strip_unsupported_latex(content, filepath.name)

    # Add section header - separate chapter number from title so anchors work
    if section_num == 0:
        header = f"# {section_title}\n\n"
    else:
        header = f"\n\n---\n\n**Chapter {section_num}**\n\n# {section_title}\n\n"

    result = header + content.strip()

    # Append footnote definitions at the end of the section
    if footnotes:
        result += "\n\n"
        for fn_id, fn_text in footnotes:
            result += f"[^{fn_id}]: {fn_text}\n"

    return result


def main():
    # Base directory is parent of this script's directory (bin/../)
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent

    combined_content = []

    # Add a title
    combined_content.append("# How to Scale Your Model: A Systems View of LLMs on TPUs\n")
    combined_content.append("*A comprehensive guide to training and serving large language models*\n\n")

    # Add table of contents - link to title anchors (not chapter-prefixed)
    combined_content.append("## Table of Contents\n\n")
    for filename, title, num in SECTION_ORDER:
        if num == 0:
            combined_content.append(f"- [{title}](#{slugify(title)})\n")
        else:
            combined_content.append(f"- [Chapter {num}: {title}](#{slugify(title)})\n")
    combined_content.append("\n")

    # Process each file
    for filename, title, num in SECTION_ORDER:
        filepath = base_dir / filename
        if filepath.exists():
            print(f"Processing {filename}...")
            content = process_file(filepath, title, num)
            combined_content.append(content)
        else:
            print(f"Warning: {filename} not found")

    # Write the combined file
    output_path = base_dir / "scaling-book-combined.md"
    output_path.write_text("\n".join(combined_content),encoding='utf-8')
    print(f"\nWritten combined file to: {output_path}")
    print(f"Total size: {output_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
