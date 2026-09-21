export function isExcluded(name: string, extra: string[] = []): boolean {
  const path = name.replaceAll('\\', '/').toLowerCase();
  return /(?:^|\/)(?:\.env[^/]*|\.git|node_modules|\.venv|venv|\.ssh|\.aws|dist|build)(?:\/|$)/.test(path)
    || /\.(?:pem|key|p12|pfx|crt|lock)$/.test(path)
    || extra.some(part => part.trim().length > 0 && path.includes(part.toLowerCase()));
}

export function redact(text: string): string {
  return text.replace(/\b(?:vck_|sk-|hf_|ghp_)[A-Za-z0-9_-]{12,}\b/g, '[REDACTED]')
    .replace(/((?:api[_-]?key|secret|password|access[_-]?token)\s*[:=]\s*["'])([^"'\n]+)(["'])/gi, '$1[REDACTED]$3');
}
