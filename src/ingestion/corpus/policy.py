"""Deterministic, explainable UET corpus rules on URL/title; no network or deletion."""
import re
import unicodedata
from urllib.parse import urlsplit, unquote
from ..identity import canonicalize_url


def normalize(text):
    text = unicodedata.normalize('NFKD', str(text).casefold().replace('đ', 'd'))
    text = ''.join(c for c in text if not unicodedata.combining(c))
    return re.sub(r'[^a-z0-9]+', ' ', text).strip()


def load_policy(path):
    import yaml
    class UniqueLoader(yaml.SafeLoader):
        pass
    def mapping(loader, node, deep=False):
        result = {}
        for key, value in node.value:
            k = loader.construct_object(key, deep=deep)
            if k in result:
                raise ValueError(f'Duplicate YAML key: {k}')
            result[k] = loader.construct_object(value, deep=deep)
        return result
    UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    policy = yaml.load(path.read_text(encoding='utf-8'), Loader=UniqueLoader)
    if not isinstance(policy, dict) or not {'version','host','rules','overrides','old_notice_through'} <= policy.keys():
        raise ValueError('Invalid corpus policy')
    if not isinstance(policy['old_notice_through'], int):
        raise ValueError('old_notice_through must be an integer')
    overrides = {}
    for url, spec in policy['overrides'].items():
        url = canonicalize_url(url)
        if urlsplit(url).hostname not in policy.get('hosts', [policy['host']]) or url in overrides:
            raise ValueError('Invalid/duplicate override URL')
        validate_spec(spec, {'include','exclude','review'})
        overrides[url] = spec
    policy['overrides'] = overrides
    ids = set()
    for rule in policy['rules']:
        validate_spec(rule, {'include','exclude','review','temporal'})
        if rule['id'] in ids:
            raise ValueError('Duplicate corpus rule id')
        ids.add(rule['id'])
        rule['_regex'] = [re.compile(p) for p in rule['patterns']]
    return policy


def validate_spec(spec, choices):
    if not isinstance(spec, dict) or spec.get('decision') not in choices:
        raise ValueError('Invalid corpus decision')
    priority = spec.get('priority', 0)
    if type(priority) is not int or not 0 <= priority <= 100:
        raise ValueError('priority must be 0..100')
    if not spec.get('reason') or not spec.get('category'):
        raise ValueError('Each rule/override needs reason and category')


def classify(row, policy):
    meta = row.get('metadata', {})
    url = row.get('uri') or meta.get('source_uri') or meta.get('final_url') or ''
    def verdict(decision, rule_id, category, reason, priority=0, **extra):
        return dict(decision=decision, rule_id=rule_id, category=category,
                    reason=reason, priority=priority, policy_version=policy['version'], **extra)
    try:
        url = canonicalize_url(url)
    except ValueError:
        return verdict('exclude','invalid_url','invalid','URL không hợp lệ')
    parts = urlsplit(url)
    if parts.hostname not in policy.get('hosts', [policy['host']]):
        return verdict('exclude','outside_site','outside_scope','Không thuộc domain site được cấu hình')
    if url in policy['overrides']:
        return verdict(rule_id='manual_override', **policy['overrides'][url])
    path = unquote(parts.path).lower()
    prefixes = policy.get('prefetch', {}).get('exclude_prefixes', [])
    if any(path.rstrip('/') == p.rstrip('/') or path.startswith(p.rstrip('/') + '/') for p in prefixes):
        return verdict('exclude','prefetch_prefix_guard','navigation','URL thuộc nhóm chặn trước HTTP')
    if re.search(r'^/(en|cdu-en|tag|author|feed|wp-admin|wp-json|login)(/|$|[’”"\'])|^/~', path):
        return verdict('exclude','non_corpus_path','navigation','Trang hệ thống, đăng nhập, lưu trữ, tiếng Anh hoặc cá nhân')
    host_allowed = policy.get('host_Allowed_only', {})
    if parts.hostname in host_allowed:
        allowed_paths = {p.rstrip('/').lower() or '/' for p in host_allowed[parts.hostname]}
        if (path.rstrip('/') or '/') not in allowed_paths:
            return verdict('exclude','host_endpoint_excluded','outside_scope',
                           f'Endpoint trên {parts.hostname} ngoài danh sách chỉ định bị loại bỏ (rác/test)')
    if meta.get('decision') == 'listing' or re.search(r'^/category/|/page/\d+/?$',path) or path in {'/','/dao-tao/','/chuong-trinh-dao-tao/'}:
        return verdict('exclude','navigation_only','navigation','Chỉ dùng khám phá link; không phải bài chi tiết')
    if meta.get('decision') == 'alias':
        return verdict('exclude','crawl_alias','crawl','URL trùng lặp (redirect alias) ở bước crawl')
    is_core_accepted = meta.get('decision') == 'accepted'
    if (meta.get('resource_type') in {'pdf','office','asset'} and not is_core_accepted) or meta.get('decision')=='deferred':
        return verdict('review','deferred_resource','attachment','Tài nguyên chưa nằm trong danh sách chỉ định; chờ xem xét',30)
    title = row.get('title') or meta.get('title') or meta.get('link_text') or ''
    texts = (normalize(title), normalize(path))
    for rule in policy['rules']:
        if not any(p.search(text) for p in rule['_regex'] for text in texts):
            continue
        decision, reason = rule['decision'], rule['reason']
        years = []
        if decision == 'include' and (not is_core_accepted or not (row.get('title') or meta.get('title') or '').strip()):
            decision = 'review'
            reason = 'URL phù hợp chủ đề nhưng chưa nằm trong danh sách core hoặc chưa tải; chờ xem xét'
        if decision == 'temporal':
            # Also catches concatenated dates such as 92020; does not interpret -2 as year.
            years = sorted({int(y) for y in re.findall(r'(?:19|20)\d{2}', title+' '+path)})
            if years and max(years) <= policy['old_notice_through']:
                decision = 'exclude'; reason = f'Thông báo theo đợt, các năm nhận diện đều ≤ {policy["old_notice_through"]}'
            else:
                decision = 'review'  # Recent publication does not prove an open deadline.
        return verdict(decision, rule['id'],rule['category'],reason,rule.get('priority',0),
                       detected_years=years, evidence='title_and_url')
    return verdict('review','unclassified','khac','Chưa nằm trong danh sách chắc chắn oke hay chắc chắn loại; chờ xem xét',20)
