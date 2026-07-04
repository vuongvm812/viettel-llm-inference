# app-config Review Checklist

Crate: `services/crates/app-config/` — YAML config parsing via `serde_yaml`, loaded at startup.

## Secrets & Credentials

- [ ] No API keys, private keys, or passwords hardcoded in source or config files committed to repo
- [ ] Secrets loaded from env vars, not YAML files (YAML may be committed; env vars are not)
- [ ] No `println!` / `log::info!` that would dump secret values at startup
- [ ] Config struct fields for secrets use `#[serde(skip_serializing)]` or a `Secret<T>` wrapper to prevent accidental logging/serialisation

## Config Schema & Validation

- [ ] Required fields are non-`Option<T>` — missing config fails fast at startup, not silently at runtime
- [ ] Numeric bounds validated at parse time (e.g., position limits > 0, fee rates in [0, 1))
- [ ] No `unwrap()` / `expect()` in config parsing — errors should propagate as `Result` with context
- [ ] Config structs derive `Debug` but sensitive fields are redacted (custom `Debug` impl or `secrecy` crate)

## Environment Separation

- [ ] Prod / staging / backtest environments are clearly separated — no `if hostname == "prod"` hacks
- [ ] Exchange endpoints (REST URL, WS URL) are configurable, not hardcoded to mainnet
- [ ] Backtest config and live trading config use distinct types or validated fields that prevent cross-contamination
- [ ] No `cfg!(debug_assertions)` branching in config logic — use explicit env flags instead

## Structural

- [ ] Config is loaded once at startup and passed as owned/shared references — not re-read mid-run
- [ ] Config struct implements `Clone` only if actually needed (avoid cloning large configs frequently)
- [ ] All public fields or accessors are documented (`///` doc comments)
- [ ] Deserialization failures produce human-readable error messages with field names and expected types
