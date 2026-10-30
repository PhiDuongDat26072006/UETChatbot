"""Join selected source IDs to processed records, preserving original payloads."""
import json
from pathlib import Path


def prepare_export(sources, path):
    chosen = {r['source_id']: r for r in sources
              if r['metadata']['corpus']['decision'] == 'include'
              and r['metadata'].get('decision') == 'accepted'}
    rows, seen, exported = [], set(), set()
    empty = 0
    for number, line in enumerate(Path(path).read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        sid = row.get('metadata', {}).get('source_id') or row.get('id')
        if not sid or sid in seen:
            raise ValueError(f'Missing/duplicate processed source ID: {path}:{number}')
        seen.add(sid)
        if sid not in chosen:
            continue
        if not isinstance(row.get('content'), str) or not row['content'].strip():
            empty += 1
            continue
        row.setdefault('metadata', {})['corpus'] = chosen[sid]['metadata']['corpus']
        rows.append(row)
        exported.add(sid)
    rows.sort(key=lambda r: r['metadata']['corpus']['rank'])
    return rows, {'exported': len(rows), 'empty': empty,
                  'missing_processed': len(set(chosen) - seen),
                  'not_exported': len(set(chosen) - exported)}
