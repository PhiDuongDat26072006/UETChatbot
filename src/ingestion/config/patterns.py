"""Các mẫu regex loại bỏ URL rác và trích xuất liên kết trang web."""
from __future__ import annotations

import re

# Các mẫu đường dẫn hệ thống WordPress, API, trang chức năng, tài khoản, cache cần loại bỏ
IGNORED_PATH_PATTERNS = [
    r'/wp-json(/|$)',
    r'/wp-includes(/|$)',
    r'/wp-content(/|$)',
    r'/xmlrpc\.php',
    r'/feed(/|$)',
    r'/comments/feed(/|$)',
    r'/trackback(/|$)',
    r'/site/setLang(/|$)',
    r'/setLang(/|$)',
    r'/language(/|$)',
    r'/login(/|$)',
    r'/logout(/|$)',
    r'/lostpassword(/|$)',
    r'/forgot-password(/|$)',
    r'/reset-password(/|$)',
    r'/resetpass(/|$)',
    r'/user-login(/|$)',
    r'/user-register(/|$)',
    r'/register(/|$)',
    r'/tai-khoan(/|$)',
    r'/account(/|$)',
    r'/user-account(/|$)',
    r'/your-profile(/|$)',
    r'/lp-checkout(/|$)',
    r'/lp-profile(/|$)',
    r'/cart(/|$)',
    r'/checkout(/|$)',
    r'/woocommerce-shop(/|$)',
    r'/post-carousel(/|$)',
    r'/pricing-table(/|$)',
    r'/university-course-carousel(/|$)',
    r'/university-event-grid(/|$)',
    r'/university-revolution-slider(/|$)',
    r'/video-banner(/|$)',
    r'/video(/|$)',
    r'/Ly[0-9A-Za-z+/=]{10,}',
]

IGNORED_PATH_REGEX = re.compile('|'.join(IGNORED_PATH_PATTERNS), re.IGNORECASE)
A_HREF_REGEX = re.compile(r'<a\b[^>]*?\bhref=["\']([^"\'#\s>]+)', re.IGNORECASE)
EMBED_SRC_REGEX = re.compile(r'<(?:embed|iframe)\b[^>]*?\bsrc=["\']([^"\'#\s>]+)', re.IGNORECASE)
