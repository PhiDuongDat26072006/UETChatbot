"""Rank existing UET sources without changing crawl/extraction state."""
import csv
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from .policy import classify, load_policy


def run_filter(args):
    from copy import copy
    import re
    from .gate import policy_paths
    if (args.data_dir / 'ingestion_state/writer.lock').exists():
        raise ValueError('Ingestion writer active; run filter/export after it finishes')
    sites = getattr(args, 'sites', None) or ['UET']
    if len(sites) != len(set(sites)):
        raise ValueError('Duplicate sites')
    if len(sites) > 1 and (args.sources or args.policy):
        raise ValueError('--sources/--policy require a single site')
    paths = policy_paths(getattr(args, 'corpus_config', None))
    prepared = []
    for site in sites:
        if not re.fullmatch(r'[A-Z0-9_-]+', site):
            raise ValueError('Invalid site')
        local = copy(args)
        local.site = site
        local.policy = args.policy or paths.get(site)
        if not local.policy:
            raise ValueError(f'{site}: add policy to corpus registry first')
        local.sources = args.sources or args.data_dir / 'data_source' / (site + '.jsonl')
        local.output_dir = ((args.output_dir / site if len(sites) > 1 else args.output_dir)
                            if args.output_dir else args.data_dir / 'corpus' / site)
        load_policy(local.policy)
        if not local.sources.is_file():
            raise ValueError(f'Missing sources: {local.sources}')
        if getattr(args, 'export_processed', False) and not (args.data_dir / 'processed_data' / (site + '.jsonl')).is_file():
            raise ValueError(f'Missing processed data for {site}')
        prepared.append(local)
    return {'status': 'success', 'sites': {a.site: run_site_filter(a) for a in prepared}}


def run_site_filter(args):
    site = args.site
    source = (args.sources or args.data_dir / f'data_source/{site}.jsonl').resolve()
    destination = (args.output_dir or args.data_dir / f'corpus/{site}').resolve()
    # Never place reports in a live ingestion stage or overwrite an input.
    if destination == source.parent or any(
        part in {'data_source', 'raw_data', 'processed_data', 'ingestion_state'}
        for part in destination.parts
    ):
        raise ValueError('Use a separate output directory, e.g. data/corpus/UET')
    policy = load_policy(args.policy)
    payload = source.read_bytes()
    records, ids = [], set()
    for line_number, line in enumerate(payload.decode('utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not isinstance(row.get('metadata', {}), dict):
            raise ValueError(f'Invalid source record at line {line_number}')
        identifier = row.get('source_id')
        if not isinstance(identifier, str) or not identifier or identifier in ids:
            raise ValueError(f'Missing/duplicate source_id at line {line_number}')
        if not isinstance(row.get('uri'), str):
            raise ValueError(f'Missing URI at line {line_number}')
        ids.add(identifier)
        row.setdefault('metadata', {})['corpus'] = classify(row, policy)
        records.append(row)
    order = {'include': 0, 'review': 1, 'exclude': 2}
    records.sort(key=lambda r: (
        order[r['metadata']['corpus']['decision']],
        -r['metadata']['corpus']['priority'], r['uri'], r['source_id']))
    for rank, row in enumerate(records, 1):
        row['metadata']['corpus']['rank'] = rank
    counts = Counter(r['metadata']['corpus']['decision'] for r in records)
    summary = {
        'status': 'success', 'policy_version': policy['version'],
        'source_sha256': hashlib.sha256(payload).hexdigest(),
        'policy_sha256': hashlib.sha256(args.policy.read_bytes()).hexdigest(),
        'records': len(records), 'decisions': dict(counts),
        'rules': dict(Counter(r['metadata']['corpus']['rule_id'] for r in records)),
        'output_dir': str(destination),
        'note': 'Title/URL heuristic only; include does not certify current validity. No pipeline state changed.',
    }
    export_rows = None
    if getattr(args, 'export_processed', False):
        from .export import prepare_export
        export_rows, summary['processed'] = prepare_export(
            records, args.data_dir / 'processed_data' / (site + '.jsonl'))
    destination.mkdir(parents=True, exist_ok=True)
    # Stage all files before replacing outputs. Each individual replacement is atomic.
    with tempfile.TemporaryDirectory(dir=destination, prefix='.filter-') as temporary:
        root = Path(temporary)
        with (root / f'{site}-ranked.jsonl').open('w', encoding='utf-8') as stream:
            for row in records:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        with (root / f'{site}-ranked.tsv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream, delimiter='\t')
            writer.writerow(['rank', 'decision', 'priority', 'category', 'title', 'url', 'reason'])
            for row in records:
                meta = row['metadata']; result = meta['corpus']
                values = [result['rank'], result['decision'], result['priority'],
                          result['category'], meta.get('title', ''), row['uri'], result['reason']]
                # Safe when opened in a spreadsheet; JSONL retains original text.
                writer.writerow(["'" + v if isinstance(v, str) and v.startswith(('=', '+', '-', '@'))
                                 else v for v in values])
        for decision in order:
            with (root / f'{site}-{decision}-links.txt').open('w', encoding='utf-8') as stream:
                for row in records:
                    if row['metadata']['corpus']['decision'] == decision:
                        stream.write(row['uri'] + '\n')
        if export_rows is not None:
            (root / 'processed').mkdir()
            with (root / 'processed' / (site + '.jsonl')).open('w', encoding='utf-8') as stream:
                for row in export_rows:
                    stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        (root / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        for staged in sorted(root.rglob('*')):
            if staged.is_file():
                target = destination / staged.relative_to(root)
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staged, target)
    return summary
