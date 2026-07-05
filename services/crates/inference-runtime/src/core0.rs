//! Core 0 — Web I/O & streaming (P1).
//!
//! tokio current-thread runtime + axum. Ingress: validate an OpenAI request,
//! claim a slab slot, write the prompt, publish `{slot, New}` on R1. Egress: a
//! cooperative task drains R4 and streams each detokenized byte as an SSE chunk;
//! on `Finish` it returns the slot to the free-list. Event-driven, never busy-spin.
//! See `design/web-io/`.
//!
//! Core-0-owned state (free-list + egress `conn` handles) lives here, not in the
//! slab: only Core 0 ever touches it, so the latency-critical cores 1/2 keep
//! `tokio` out of their storage.

use crate::affinity;
use crate::config::Config;
use crate::pipeline::{EgressPoller, IngressProducer, Pipeline};
use crate::rings::{EventKind, RingEvent};
use crate::slab::Slab;
use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, State};
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::Router;
use disruptor::{Polling, Producer};
use serde::Deserialize;
use std::convert::Infallible;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;
use tokio::sync::mpsc::{unbounded_channel, UnboundedSender};
use tokio_stream::wrappers::UnboundedReceiverStream;
use tokio_stream::StreamExt;

/// Sender half of a request's SSE body. Core 0 ingress registers it in `conns`;
/// Core 0 egress takes it on `Finish` (dropping it ends the stream).
pub type EgressHandle = UnboundedSender<Egress>;

/// What the egress poller pushes toward a connection.
pub enum Egress {
    Chunk(String),
    Done,
}

/// Upper bound on `max_tokens` accepted at the trust boundary.
const MAX_TOKENS_CAP: u32 = 8192;
/// Explicit request-body cap at the trust boundary (trace prompts are ~80 KB).
const MAX_BODY_BYTES: usize = 4 * 1024 * 1024;

type Free = Arc<Mutex<Vec<u32>>>;
type Conns = Arc<Mutex<Vec<Option<EgressHandle>>>>;
/// Per-slot read cursor into the slab's `out_bytes`: how many detok bytes Core 0
/// has already streamed. Reset on claim. Only Core 0 touches it.
type Cursors = Arc<Mutex<Vec<usize>>>;

/// Recover from mutex poisoning instead of cascading a panic to every later
/// request — one panicking handler must not wedge the whole server.
fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

#[derive(Clone)]
struct AppState {
    slab: Arc<Slab>,
    free: Free,
    conns: Conns,
    cursor: Cursors,
    ingress: Arc<Mutex<IngressProducer>>,
}

impl AppState {
    fn new(slab: Arc<Slab>, ingress: IngressProducer) -> Self {
        let cap = slab.capacity();
        AppState {
            slab,
            // Only Core 0 claims (ingress) / returns (egress) slots, so a plain
            // Vec behind one Mutex suffices — no Treiber stack, no atomics.
            free: Arc::new(Mutex::new((0..cap as u32).rev().collect())),
            conns: Arc::new(Mutex::new((0..cap).map(|_| None).collect())),
            cursor: Arc::new(Mutex::new(vec![0; cap])),
            ingress: Arc::new(Mutex::new(ingress)),
        }
    }
}

#[derive(Deserialize)]
struct ChatRequest {
    #[serde(default)]
    model: String,
    #[serde(default)]
    messages: Vec<ChatMessage>,
    max_tokens: Option<u32>,
    // Carried through for Core 2 determinism in later phases; unused by the mock.
    #[allow(dead_code)]
    temperature: Option<f32>,
    #[allow(dead_code)]
    seed: Option<u64>,
}

#[derive(Deserialize)]
struct ChatMessage {
    #[allow(dead_code)]
    role: String,
    #[serde(default)]
    content: String,
}

/// Run Core 0: the HTTP server plus the R4 egress drain. Serves until Ctrl-C /
/// SIGINT, then shuts down cleanly, propagating I/O errors to `main`.
pub fn serve(cfg: &Config, slab: Arc<Slab>, pipeline: Pipeline) -> std::io::Result<()> {
    serve_until(cfg, slab, pipeline, shutdown_signal())
}

/// Like `serve`, but stops when `shutdown` resolves: axum stops accepting, drains
/// in-flight connections, and returns `Ok`. The seam is load-bearing for P6 —
/// LLVM writes PGO `*.profraw` only via a libc `atexit` handler that a killed
/// process never runs, so the build must reach a clean exit. It also lets tests
/// trigger shutdown deterministically. `shutdown` is `Send + 'static` per axum.
/// Crate-internal (bin crate has no public API) — `pub(crate)` states the intent.
pub(crate) fn serve_until(
    cfg: &Config,
    slab: Arc<Slab>,
    pipeline: Pipeline,
    shutdown: impl std::future::Future<Output = ()> + Send + 'static,
) -> std::io::Result<()> {
    affinity::pin(cfg.runtime.cores.web_io); // no-op on macOS

    let Pipeline {
        ingress,
        egress,
        core1: _core1,
        core2: _core2,
    } = pipeline;
    let state = AppState::new(slab, ingress);

    let addr = format!("{}:{}", cfg.server.host, cfg.server.port);
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let local = tokio::task::LocalSet::new();
    local.block_on(&rt, async move {
        let listener = tokio::net::TcpListener::bind(&addr).await?;
        println!("inference-runtime listening on http://{addr} (P1 mock backend)");

        tokio::task::spawn_local(egress_loop(
            Arc::clone(&state.slab),
            Arc::clone(&state.conns),
            Arc::clone(&state.free),
            Arc::clone(&state.cursor),
            egress,
        ));
        axum::serve(listener, build_app(state))
            .with_graceful_shutdown(shutdown)
            .await?;
        Ok(())
    })
}

/// Resolve on Ctrl-C / SIGINT (what the PGO script sends) or SIGTERM (systemd's
/// default stop signal on the deploy box) so `serve` returns and `main` exits
/// cleanly — required for the PGO profile flush (see `serve_until`) and for a
/// graceful production stop.
async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let term = async {
        use tokio::signal::unix::{signal, SignalKind};
        match signal(SignalKind::terminate()) {
            Ok(mut s) => {
                s.recv().await;
            }
            // No SIGTERM stream (should not happen on unix) → fall back to SIGINT only.
            Err(_) => std::future::pending::<()>().await,
        }
    };
    #[cfg(not(unix))]
    let term = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => {}
        _ = term => {}
    }
}

/// Build the router. Shared by `serve` and the HTTP tests.
fn build_app(state: AppState) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat))
        .route("/health", get(|| async { "ok" }))
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .with_state(state)
}

/// Cooperatively drain R4 and stream bytes to the right connection.
async fn egress_loop(slab: Arc<Slab>, conns: Conns, free: Free, cursor: Cursors, mut r4: EgressPoller) {
    loop {
        match r4.poll() {
            Ok(mut guard) => {
                for ev in &mut guard {
                    let idx = ev.slot as usize;
                    match ev.kind {
                        EventKind::Piece(delta) => {
                            // Piece carries the count of newly-committed complete-
                            // UTF-8 bytes; read them from the slab's out_bytes at
                            // our cursor. Core 1 appended and published these bytes
                            // before this event (R4 release/acquire), so the read is
                            // of settled, disjoint bytes. `read_committed` builds the
                            // slice from a raw base — no reference into a cell Core 1
                            // may write — and bounds-checks. Bytes are already complete
                            // UTF-8, so from_utf8_lossy is belt-and-suspenders.
                            // Advance the cursor in the same lock scope (one acquire).
                            let start = {
                                let mut c = lock(&cursor);
                                let s = c[idx];
                                c[idx] = s + delta as usize;
                                s
                            };
                            // SAFETY: [start, start+delta) is a range Core 1 committed
                            // and published; read_committed bounds-checks the rest.
                            match unsafe { slab.read_committed(ev.slot, start, delta as usize) } {
                                Ok(bytes) => {
                                    let piece = String::from_utf8_lossy(bytes).into_owned();
                                    if let Some(tx) = lock(&conns)[idx].as_ref() {
                                        let _ = tx.send(Egress::Chunk(piece));
                                    }
                                }
                                // A bounds error is a logic bug, not client input —
                                // drop the chunk (don't UB) and keep the stream alive.
                                Err(e) => eprintln!("core0 egress read (slot {}): {e}", ev.slot),
                            }
                        }
                        EventKind::Finish(_) => {
                            if let Some(tx) = lock(&conns)[idx].take() {
                                let _ = tx.send(Egress::Done);
                            }
                            lock(&free).push(ev.slot);
                        }
                        EventKind::Token(_) | EventKind::New => {} // R3/R1 kinds; never on R4
                    }
                }
                // Yield so ingress handlers on this single thread get a turn.
                tokio::task::yield_now().await;
            }
            // Idle: park briefly instead of hot-looping the async executor.
            Err(Polling::NoEvents) => tokio::time::sleep(Duration::from_micros(200)).await,
            Err(Polling::Shutdown) => break,
        }
    }
}

async fn chat(State(app): State<AppState>, body: Bytes) -> Response {
    // Parse the request body with sonic-rs (SIMD JSON) rather than the serde_json-backed
    // axum `Json` extractor. `Bytes` still honours `DefaultBodyLimit`; a malformed body
    // is a 400 at the trust boundary.
    let req: ChatRequest = match sonic_rs::from_slice(body.as_ref()) {
        Ok(req) => req,
        Err(_) => return bad_request("invalid JSON body"),
    };

    // Validate at the trust boundary before burning a slot.
    if req.model.trim().is_empty() {
        return bad_request("`model` is required");
    }
    if req.messages.is_empty() {
        return bad_request("`messages` must not be empty");
    }
    let max_tokens = req.max_tokens.unwrap_or(128);
    if max_tokens == 0 || max_tokens > MAX_TOKENS_CAP {
        return bad_request("`max_tokens` out of range");
    }

    // Claim a slot (admission cap = slab capacity).
    let slot = match lock(&app.free).pop() {
        Some(s) => s,
        None => return capacity_503("server at capacity"),
    };

    let (tx, rx) = unbounded_channel::<Egress>();

    // Write the slot directly. Core 0 owns the ingress stage until the R1 publish,
    // and the prompt is assembled once into the pre-reserved buffer — no temp
    // String/Vec, no second copy (keeps the slab's "written once" invariant).
    // SAFETY: slot just came off the free-list; no other core holds it yet.
    {
        let s = unsafe { app.slab.slot_mut(slot) };
        s.reset();
        for (i, m) in req.messages.iter().enumerate() {
            if i > 0 {
                s.prompt.push('\n');
            }
            s.prompt.push_str(&m.content);
        }
        s.max_tokens = max_tokens;
    }
    // Reset the egress read cursor for the reused slot: `s.reset()` cleared
    // out_bytes/out_committed, so Core 0 must stream this request from offset 0.
    // Without this the cursor keeps a prior request's byte count → stale reads and,
    // once it exceeds out_bytes capacity, an out-of-bounds read.
    lock(&app.cursor)[slot as usize] = 0;
    lock(&app.conns)[slot as usize] = Some(tx);

    // `try_publish`, never a blocking `publish`: this runs on Core 0's single
    // async thread, and a full-ring spin would starve the egress drain (and every
    // other connection) → deadlock. With `ring_size >= max_inflight` (enforced in
    // pipeline::spawn) R1 can't fill, so this is belt-and-suspenders.
    let published = lock(&app.ingress)
        .try_publish(|e| {
            *e = RingEvent {
                slot,
                kind: EventKind::New,
            }
        })
        .is_ok();
    if !published {
        lock(&app.conns)[slot as usize] = None; // drop tx
        lock(&app.free).push(slot);
        return capacity_503("ingress ring full");
    }

    // ponytail: always SSE for P1 (the exit criterion). `stream:false` buffering
    // lands when the benchmark harness needs it (P5).
    let stream = UnboundedReceiverStream::new(rx).map(|egress| {
        Ok::<Event, Infallible>(match egress {
            Egress::Chunk(text) => Event::default().data(sse_chunk(&text)),
            Egress::Done => Event::default().data("[DONE]"),
        })
    });
    Sse::new(stream).into_response()
}

/// Minimal OpenAI-shaped streaming delta so `curl` output reads as chat completions.
fn sse_chunk(text: &str) -> String {
    let content = sonic_rs::to_string(text).unwrap_or_else(|_| "\"\"".into());
    format!(
        r#"{{"object":"chat.completion.chunk","choices":[{{"index":0,"delta":{{"content":{content}}}}}]}}"#
    )
}

fn bad_request(msg: &'static str) -> Response {
    (axum::http::StatusCode::BAD_REQUEST, msg).into_response()
}

fn capacity_503(msg: &'static str) -> Response {
    (axum::http::StatusCode::SERVICE_UNAVAILABLE, msg).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CANNED_REPLY;
    use crate::pipeline;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use sonic_rs::JsonValueTrait; // `.as_str()` on sonic_rs::Value
    use tower::ServiceExt; // oneshot

    fn test_cfg(cap: u32) -> Config {
        let yaml = format!(
            r#"
server: {{ host: "127.0.0.1", port: 0 }}
model: {{ gguf_path: "x", n_ctx: 1024, n_threads: 1, n_gpu_layers: -1 }}
runtime:
  max_inflight: {cap}
  ring_size: 1024
  cores: {{ web_io: 0, text: 1, fast_loop: 2 }}
"#
        );
        serde_yaml::from_str(&yaml).expect("cfg")
    }

    fn post(body: &str) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(body.to_owned()))
            .unwrap()
    }

    async fn status_of(state: &AppState, body: &str) -> StatusCode {
        build_app(state.clone()).oneshot(post(body)).await.unwrap().status()
    }

    #[test]
    fn sse_chunk_is_a_valid_openai_delta() {
        // Content with a quote must be JSON-escaped, not corrupt the frame.
        let frame = sse_chunk("a\"b");
        let v: sonic_rs::Value = sonic_rs::from_str(&frame).expect("valid json frame");
        assert_eq!(v["object"].as_str(), Some("chat.completion.chunk"));
        assert_eq!(v["choices"][0]["delta"]["content"].as_str(), Some("a\"b"));
    }

    /// End-to-end HTTP: a real OpenAI request streams SSE `chat.completion.chunk`
    /// frames through all four rings, truncates at `max_tokens`, ends with
    /// `[DONE]`, and the slot is recycled. Also covers 400 (validation) and 503
    /// (capacity). This is the exit criterion's "curl → SSE" surface.
    #[test]
    fn http_streams_sse_validates_and_recycles() {
        let cfg = test_cfg(2);
        let slab = Arc::new(Slab::new(2, 128, 128, 256));
        let Pipeline {
            ingress,
            egress,
            core1,
            core2,
        } = pipeline::spawn(&cfg, Arc::clone(&slab));
        let state = AppState::new(slab, ingress);

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let local = tokio::task::LocalSet::new();
        local.block_on(&rt, async {
            tokio::task::spawn_local(egress_loop(
                Arc::clone(&state.slab),
                Arc::clone(&state.conns),
                Arc::clone(&state.free),
                Arc::clone(&state.cursor),
                egress,
            ));

            // Happy path with max_tokens=5 → exactly 5 content chunks + [DONE].
            let ok = r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"max_tokens":5}"#;
            let resp = build_app(state.clone()).oneshot(post(ok)).await.unwrap();
            assert_eq!(resp.status(), StatusCode::OK);
            let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
            let text = String::from_utf8_lossy(&bytes);
            assert_eq!(
                text.matches("chat.completion.chunk").count(),
                5,
                "max_tokens=5 → 5 content frames (MaxTokens truncation)"
            );
            let first = CANNED_REPLY[0] as char; // 'H'
            assert!(text.contains(&format!(r#""content":"{first}""#)), "streamed content");
            assert!(text.contains("[DONE]"), "terminal frame");

            // Slot recycled: after the stream finished, capacity is back.
            assert_eq!(lock(&state.free).len(), 2, "slot returned to free-list");

            // 400: empty model, empty messages, bad max_tokens.
            assert_eq!(
                status_of(&state, r#"{"model":"","messages":[{"role":"u","content":"x"}]}"#).await,
                StatusCode::BAD_REQUEST
            );
            assert_eq!(
                status_of(&state, r#"{"model":"m","messages":[]}"#).await,
                StatusCode::BAD_REQUEST
            );
            assert_eq!(
                status_of(&state, r#"{"model":"m","messages":[{"role":"u","content":"x"}],"max_tokens":0}"#).await,
                StatusCode::BAD_REQUEST
            );
            // Syntactically invalid JSON → 400 (the sonic-rs parse-error branch that
            // replaced axum's Json extractor).
            assert_eq!(status_of(&state, "{not valid json").await, StatusCode::BAD_REQUEST);

            // 503: exhaust the free-list, next request is rejected without a slot.
            lock(&state.free).clear();
            assert_eq!(
                status_of(&state, r#"{"model":"m","messages":[{"role":"u","content":"x"}],"max_tokens":5}"#).await,
                StatusCode::SERVICE_UNAVAILABLE
            );
        });

        // Clean shutdown: drop the producer → cascade R1→R2→R3→R4, join workers.
        drop(state);
        core1.join().expect("core1 join");
        core2.join().expect("core2 join");
    }

    /// Concatenate the `delta.content` of every `chat.completion.chunk` SSE frame.
    /// Each frame is rendered independently by Core 0 (`from_utf8_lossy` per Piece),
    /// so if the UTF-8 gate ever emitted a partial code point this would contain a
    /// U+FFFD replacement char instead of the real multi-byte character.
    fn sse_content(body: &str) -> String {
        body.lines()
            .filter_map(|l| l.strip_prefix("data:").map(str::trim))
            .filter_map(|d| sonic_rs::from_str::<sonic_rs::Value>(d).ok())
            .filter_map(|v| v["choices"][0]["delta"]["content"].as_str().map(str::to_owned))
            .collect()
    }

    /// End-to-end streaming must reconstruct the full multi-byte reply (`café ☕`)
    /// exactly — proving the detokenize handoff never splits a code point across
    /// SSE frames. Runs TWO requests through a **single-slot** server so the second
    /// reuses the slot: that catches a stale egress cursor (which would stream the
    /// first reply's leftover bytes, or read out of bounds).
    #[test]
    fn http_streams_multibyte_and_recycles_the_cursor() {
        let expected = std::str::from_utf8(CANNED_REPLY).expect("canned reply is valid utf8");
        assert!(expected.contains('☕'), "reply must exercise a 3-byte char");

        let cfg = test_cfg(1); // single slot → the 2nd request must reuse slot 0
        let slab = Arc::new(Slab::new(1, 128, 512, 256));
        let Pipeline { ingress, egress, core1, core2 } = pipeline::spawn(&cfg, Arc::clone(&slab));
        let state = AppState::new(slab, ingress);

        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        let local = tokio::task::LocalSet::new();
        local.block_on(&rt, async {
            tokio::task::spawn_local(egress_loop(
                Arc::clone(&state.slab),
                Arc::clone(&state.conns),
                Arc::clone(&state.free),
                Arc::clone(&state.cursor),
                egress,
            ));

            // max_tokens above the reply length → full reply, natural (Eos) end.
            let body = r#"{"model":"m","messages":[{"role":"u","content":"hi"}],"max_tokens":500}"#;
            for attempt in 0..2 {
                let resp = build_app(state.clone()).oneshot(post(body)).await.unwrap();
                assert_eq!(resp.status(), StatusCode::OK, "attempt {attempt}");
                let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
                let content = sse_content(&String::from_utf8_lossy(&bytes));
                assert_eq!(content, expected, "attempt {attempt}: full multi-byte reply reconstructed");
                // Slot freed → next iteration reuses slot 0 (cursor must have reset).
                assert_eq!(lock(&state.free).len(), 1, "slot recycled after attempt {attempt}");
            }
        });

        drop(state);
        core1.join().expect("core1 join");
        core2.join().expect("core2 join");
    }

    /// Graceful shutdown: once the shutdown future resolves, `serve_until` stops
    /// accepting, drains in-flight, and returns `Ok` instead of blocking forever.
    /// This is the clean-exit path the PGO build depends on — LLVM flushes
    /// `*.profraw` only via a libc `atexit` handler, which a signal-killed process
    /// never runs. Run on a thread so a regression that drops
    /// `.with_graceful_shutdown` (which blocks forever) fails as a clean timeout,
    /// not a hung test that burns the whole CI budget.
    #[test]
    fn serve_until_returns_on_shutdown() {
        let cfg = test_cfg(2); // port 0 → ephemeral bind
        let slab = Arc::new(Slab::new(2, 128, 128, 256));
        let pipe = pipeline::spawn(&cfg, Arc::clone(&slab));
        // serve_until consumes the Pipeline; when it returns it drops the ingress
        // producer, which cascades R1→R4 Shutdown and lets Core 1/Core 2 exit on
        // their own (same cascade pipeline.rs's tests join on) — the busy-spin
        // workers wind down here, they don't leak for the rest of the suite.
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let res = serve_until(&cfg, slab, pipe, async {
                tokio::time::sleep(Duration::from_millis(50)).await;
            });
            let _ = tx.send(res.is_ok());
        });
        let ok = rx
            .recv_timeout(Duration::from_secs(5))
            .expect("serve_until must return after graceful shutdown, not hang");
        assert!(ok, "serve_until must return Ok after graceful shutdown");
    }
}
