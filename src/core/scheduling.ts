export class RevisionGuard {
  private revision = 0;
  private controller?: AbortController;
  invalidate(): number { this.controller?.abort(); this.controller = undefined; return ++this.revision; }
  current(id: number): boolean { return id === this.revision; }
  signal(id: number): AbortSignal {
    if (!this.current(id)) throw new Error('Obsolete evaluation');
    this.controller?.abort();
    this.controller = new AbortController();
    return this.controller.signal;
  }
}

export class RequestBudget {
  private timestamps: number[] = [];
  private cooldownUntil = 0;
  constructor(private spacingMs = 2000) {}
  delay(limit: number, now = Date.now()): number {
    this.timestamps = this.timestamps.filter(time => now - time < 60_000);
    const rateDelay = this.timestamps.length >= limit ? (this.timestamps[0]! + 60_000 - now) : 0;
    const spacing = this.timestamps.length ? this.timestamps[this.timestamps.length - 1]! + this.spacingMs - now : 0;
    return Math.max(0, rateDelay, spacing, this.cooldownUntil - now);
  }
  record(now = Date.now()): void { this.timestamps.push(now); }
  cooldown(ms: number, now = Date.now()): void { this.cooldownUntil = now + ms; }
}

export class ReportCache<T> {
  private entries = new Map<string, { value: T; time: number }>();
  get(key: string, now = Date.now()): T | undefined {
    const entry = this.entries.get(key);
    if (!entry || now - entry.time > 120_000) { this.entries.delete(key); return; }
    return entry.value;
  }
  set(key: string, value: T, now = Date.now()): void {
    this.entries.delete(key);
    this.entries.set(key, { value, time: now });
    while (this.entries.size > 20) this.entries.delete(this.entries.keys().next().value!);
  }
  clear(): void { this.entries.clear(); }
}
