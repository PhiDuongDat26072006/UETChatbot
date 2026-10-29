"""Bounded HTML table expansion; uncertain structures remain readable and reviewable."""
import re


class TableStructureError(ValueError):
    pass


def _span(cell, key, remaining):
    value = str(cell.get(key, '1')).strip()
    if not value.isdigit():
        raise TableStructureError('invalid_span')
    value = int(value)
    if key == 'rowspan' and value == 0:
        return remaining
    if not 1 <= value <= 1000:
        raise TableStructureError('invalid_span')
    return value


def table_markdown(table, url='', render_cell=None):
    """Return Markdown; annotate DOM on structural uncertainty for quality gate."""
    if render_cell is None:
        render_cell = lambda c: c.get_text(' ', strip=True)
    rows = [r for r in table.find_all('tr') if r.find_parent('table') is table]
    def value(cell):
        text = render_cell(cell).strip()
        text = re.sub(r'\s*\n\s*', ' ; ', text)
        return text.replace('|', r'\|')
    try:
        if table.find('table') is not None:
            raise TableStructureError('nested_table')
        if len(rows) > 5000:
            raise TableStructureError('table_size_limit')
        grid, origins = {}, {}
        header_count = 0
        for ri, row in enumerate(rows):
            cells = row.find_all(['td', 'th'], recursive=False)
            group = row.parent
            remaining = sum(r.parent is group for r in rows[ri:])
            col = 0
            for cell in cells:
                while (ri, col) in grid:
                    col += 1
                rs = _span(cell, 'rowspan', remaining)
                cs = _span(cell, 'colspan', remaining)
                if rs > remaining or col + cs > 200:
                    raise TableStructureError('span_out_of_bounds')
                content = value(cell)
                if len(grid) + rs * cs > 100000:
                    raise TableStructureError('table_size_limit')
                for r in range(ri, ri + rs):
                    for c in range(col, col + cs):
                        if (r, c) in grid:
                            raise TableStructureError('overlapping_cells')
                        grid[r, c] = content
                        origins[r, c] = (ri, col)
                col += cs
            explicit = row.find_parent('thead') is not None or (cells and all(c.name == 'th' for c in cells))
            bold_first = ri == 0 and cells and all(c.find(['strong', 'b']) for c in cells)
            if ri == header_count and (explicit or bold_first):
                header_count += 1
        width = max((c + 1 for _, c in grid), default=0)
        if not width:
            return ''
        output = []
        for ri in range(len(rows)):
            vals = [grid.get((ri, c), '') for c in range(width)]
            if not any(vals):
                continue  # Empty separators only; never fill genuine empty cells.
            if any((ri, c) not in grid for c in range(width)):
                raise TableStructureError('ragged_rows')
            if ri >= header_count:
                output.append(vals)
        headers = []
        for c in range(width):
            seen, parts = set(), []
            for r in range(header_count):
                origin = origins.get((r, c))
                text = grid.get((r, c), '')
                if text and origin not in seen:
                    parts.append(text); seen.add(origin)
            headers.append(' / '.join(parts) or f'Cột {c + 1}')
        # A nonempty horizontal data merge is ambiguous: don't invent per-column facts.
        for ri in range(header_count, len(rows)):
            nonempty_origins = [origins[ri,c] for c in range(width) if grid.get((ri,c))]
            if len(nonempty_origins) != len(set(nonempty_origins)):
                raise TableStructureError('merged_data_columns')
        lines = [headers, ['---'] * width, *output]
        caption = table.find('caption', recursive=False)
        prefix = render_cell(caption).strip() + '\n\n' if caption else ''
        return prefix + '\n'.join('| ' + ' | '.join(r) + ' |' for r in lines)
    except TableStructureError as exc:
        table['data-extraction-issue'] = str(exc)
        # Preserve each original cell once with explicit coordinates and merge extent.
        lines = ['**Bảng cần kiểm tra cấu trúc**']
        caption = table.find('caption', recursive=False)
        if caption:
            lines.append(render_cell(caption).strip())
        for i, row in enumerate(rows, 1):
            cells = row.find_all(['th', 'td'], recursive=False)
            parts = []
            for j, cell in enumerate(cells, 1):
                text = value(cell)
                if text:
                    span = ', '.join(f'{k}={cell[k]}' for k in ('rowspan','colspan') if cell.has_attr(k))
                    parts.append(f'  - Ô {j}' + (f' ({span})' if span else '') + ': ' + text)
            if parts:
                lines.extend([f'- Hàng {i}:', *parts])
        return '\n'.join(lines)
