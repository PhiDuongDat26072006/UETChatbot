"""Per-run corpus policies shared by network and extraction coordinators."""
from pathlib import Path
from copy import deepcopy
from urllib.parse import urlsplit, unquote
import hashlib
import re
import yaml
from .policy import load_policy, classify, normalize

DEFAULT_CONFIG = Path(__file__).parents[1] / 'config/corpus/sites.yaml'


def policy_paths(config=None):
    path = Path(config or DEFAULT_CONFIG)
    data = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict) or data.get('version') != 1 or not isinstance(data.get('sites'), dict):
        raise ValueError('Corpus registry requires version: 1 and sites mapping')
    result = {}
    for site, name in data['sites'].items():
        if not re.fullmatch(r'[A-Z0-9_-]+', site) or not isinstance(name, str):
            raise ValueError('Invalid corpus site/path')
        result[site] = (path.parent / name).resolve()
    return result


def attach(profiles, options):
    mode = getattr(options, 'corpus_mode', 'balanced')
    paths = policy_paths(getattr(options, 'corpus_config', None)) if mode != 'off' else {}
    for site, profile in profiles.items():
        profile['_corpus_mode'] = mode
        if mode == 'off' or site not in paths:
            if mode == 'strict' and (not options.sites or site in options.sites):
                raise ValueError(f'{site}: strict mode requires a corpus policy')
            continue
        policy = load_policy(paths[site])
        if not set(policy.get('hosts', [policy['host']])) <= set(profile['scope']['hosts']):
            raise ValueError(f'{site}: corpus hosts must belong to crawl scope')
        profile['_corpus_policy'] = policy
        profile['_corpus_hash'] = hashlib.sha256(paths[site].read_bytes()).hexdigest()
        profile['_corpus_prefetch'] = [re.compile(p) for p in policy.get('prefetch', {}).get('exclude_patterns', [])]
    return profiles


def annotate(row, profile):
    if not profile.get('_corpus_policy'):
        return None
    result = classify(row, profile['_corpus_policy'])
    result['policy_hash'] = profile['_corpus_hash']
    row.setdefault('metadata', {})['corpus'] = result
    return result


def selected(row, profile):
    result = annotate(row, profile)
    return result is None or result['decision'] == 'include' or (
        result['decision'] == 'review' and profile.get('_corpus_mode') != 'strict')


def prefetch(url, profile):
    """Only explicit URL guards skip HTTP; unknowns and navigation still traverse."""
    if not profile.get('_corpus_policy'):
        return None
    from ..crawler.policy import is_listing
    from ..identity import normalize_site_url
    url = normalize_site_url(url, profile)
    manual = profile.get('_overrides', {}).get(url)
    if manual and manual['decision'] in {'accepted', 'listing'}:
        return None
    override = profile['_corpus_policy']['overrides'].get(url)
    if override:
        return 'manual_exclude' if override['decision'] == 'exclude' else None
    if url in [normalize_site_url(u, profile) for u in profile['seeds']]:
        return None
    raw_path = unquote(urlsplit(url).path).rstrip('/') or '/'
    prefixes = profile['_corpus_policy'].get('prefetch', {}).get('exclude_prefixes', [])
    if any(raw_path == p.rstrip('/') or raw_path.startswith(p.rstrip('/') + '/') for p in prefixes):
        return 'prefetch_prefix_guard'
    if is_listing(url, profile):
        return None
    path = normalize(raw_path)
    if any(p.search(path) for p in profile.get('_corpus_prefetch', [])):
        return 'prefetch_url_guard'
    return None


def priority(url, profile, metadata=None):
    if not profile.get('_corpus_policy'):
        return 0
    hint = deepcopy(metadata or {})
    hint['title'] = hint.get('title') or hint.get('link_text', '')
    result = classify({'uri': url, 'metadata': hint}, profile['_corpus_policy'])
    return result.get('priority', 0)


def reconcile(storage, site, profile):
    if not profile.get('_corpus_policy'):
        return 0
    policy = profile['_corpus_policy']
    seeds = [url for url, rule in policy['overrides'].items() if rule['decision'] == 'include']
    profile['_manual_seeds'] = list(dict.fromkeys(seeds + profile.get('_manual_seeds', [])))
    changed = 0
    for row in storage.records('sources', site).values():
        before = row['metadata'].get('corpus')
        changed += before != annotate(row, profile)
    if changed:
        storage.flush('sources', site)
    return changed
