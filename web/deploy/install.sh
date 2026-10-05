#!/usr/bin/env bash
# Runs on the VPS as deploygp, called by .github/workflows/deploy-web.yml after the release upload.
set -euo pipefail
SITE=/www/wwwroot/varbox.guestpassvms.com
REL="$HOME/varbox-release"
VHOST_REWRITE=/www/server/panel/vhost/rewrite/varbox.guestpassvms.com.conf

# deploygp owns the site folder so releases can replace files; www owns storage for PHP.
if [ "$(stat -c %U "$SITE")" != "deploygp" ]; then
  sudo -n /usr/bin/install -d -o deploygp -g www -m 2775 "$SITE"
fi
rsync -rl --delete --omit-dir-times --no-perms --no-owner --no-group \
  --exclude 'storage/' --exclude '.user.ini' --exclude '.well-known/' --exclude '.htaccess' \
  --exclude '404.html' --exclude '502.html' \
  "$REL/public/" "$SITE/"
if [ ! -d "$SITE/storage" ] || [ "$(stat -c %U "$SITE/storage")" != "www" ]; then
  sudo -n /usr/bin/install -d -o www -g www -m 2775 "$SITE/storage"
fi
if [ ! -f "$SITE/storage/config.json" ]; then
  umask 007
  python3 - "$SITE/storage/config.json" <<'PY'
import json, secrets, sys
json.dump({"worker_token": secrets.token_urlsafe(32), "setup_code": secrets.token_hex(4)}, open(sys.argv[1], "w"))
PY
  echo "created storage/config.json (worker token and one-time setup code)"
fi

# Routing rules, then a checked reload.
if ! sudo -n /usr/bin/test -f "$VHOST_REWRITE" || ! sudo -n /bin/cp "$VHOST_REWRITE" /tmp/varbox_rewrite_live.conf || ! cmp -s /tmp/varbox_rewrite_live.conf "$REL/nginx/varbox.rewrite.conf"; then
  sudo -n /bin/cp "$REL/nginx/varbox.rewrite.conf" "$VHOST_REWRITE"
  sudo -n /www/server/nginx/sbin/nginx -t
  sudo -n /www/server/nginx/sbin/nginx -s reload
  echo "nginx routing updated"
fi
echo "release $(cat "$REL/RELEASE" 2>/dev/null || echo unknown) installed"
