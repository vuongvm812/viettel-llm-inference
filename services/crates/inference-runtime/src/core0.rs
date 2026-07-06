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
        .route("/v1/ws", get(ws_upgrade)) // P7: WebSocket streaming alongside SSE
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

/// A rejected ingress at the trust boundary — the caller renders it in its own transport
/// (an HTTP status for SSE, a WS text frame + close for WebSocket).
struct IngestError {
    status: axum::http::StatusCode,
    msg: &'static str,
}

/// Shared ingress for both SSE and WebSocket (P7): parse + validate the OpenAI request,
/// claim a slab slot, write the prompt once, register the egress sender, and publish
/// `{slot, New}` on R1. Returns the egress stream to forward to the client, or an
/// [`IngestError`] for the caller to surface. Both streaming transports ride the identical
/// path — only the framing of the returned stream differs.
fn start_request(app: &AppState, body: &[u8]) -> Result<UnboundedReceiverStream<Egress>, IngestError> {
    // Body-size cap at the shared choke point. The SSE path is already bounded by axum's
    // `DefaultBodyLimit`, but that's an HTTP-body limit — it does NOT apply to a WebSocket
    // frame. Enforcing it here covers both transports (a WS client can otherwise submit a
    // ~64 MiB first frame — 16× the intended cap — to amplify parse + slab-realloc cost).
    if body.len() > MAX_BODY_BYTES {
        return Err(IngestError {
            status: axum::http::StatusCode::PAYLOAD_TOO_LARGE,
            msg: "request body too large",
        });
    }
    // Parse with sonic-rs (SIMD JSON) rather than the serde_json-backed axum `Json`
    // extractor. A malformed body is a 400 at the trust boundary.
    let req: ChatRequest = sonic_rs::from_slice(body).map_err(|_| IngestError {
        status: axum::http::StatusCode::BAD_REQUEST,
        msg: "invalid JSON body",
    })?;

    let bad = |msg| IngestError { status: axum::http::StatusCode::BAD_REQUEST, msg };
    if req.model.trim().is_empty() {
        return Err(bad("`model` is required"));
    }
    if req.messages.is_empty() {
        return Err(bad("`messages` must not be empty"));
    }
    let max_tokens = req.max_tokens.unwrap_or(128);
    if max_tokens == 0 || max_tokens > MAX_TOKENS_CAP {
        return Err(bad("`max_tokens` out of range"));
    }

    // Claim a slot (admission cap = slab capacity).
    let slot = lock(&app.free).pop().ok_or(IngestError {
        status: axum::http::StatusCode::SERVICE_UNAVAILABLE,
        msg: "server at capacity",
    })?;

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
    lock(&app.cursor)[slot as usize] = 0;
    lock(&app.conns)[slot as usize] = Some(tx);

    // `try_publish`, never a blocking `publish`: this runs on Core 0's single async
    // thread, and a full-ring spin would starve the egress drain → deadlock. With
    // `ring_size >= max_inflight` (enforced in pipeline::spawn) R1 can't fill.
    let published = lock(&app.ingress)
        .try_publish(|e| *e = RingEvent { slot, kind: EventKind::New })
        .is_ok();
    if !published {
        lock(&app.conns)[slot as usize] = None; // drop tx
        lock(&app.free).push(slot);
        return Err(IngestError {
            status: axum::http::StatusCode::SERVICE_UNAVAILABLE,
            msg: "ingress ring full",
        });
    }
    Ok(UnboundedReceiverStream::new(rx))
}

/// SSE endpoint (OpenAI `stream: true`). Thin wrapper over the shared [`start_request`].
///
/// ponytail: always SSE (no `stream:false` buffering yet — lands with the benchmark harness).
async fn chat(State(app): State<AppState>, body: Bytes) -> Response {
    let rx = match start_request(&app, body.as_ref()) {
        Ok(rx) => rx,
        Err(e) => return (e.status, e.msg).into_response(),
    };
    let stream = rx.map(|egress| {
        Ok::<Event, Infallible>(match egress {
            Egress::Chunk(text) => Event::default().data(sse_chunk(&text)),
            Egress::Done => Event::default().data("[DONE]"),
        })
    });
    Sse::new(stream).into_response()
}

/// WebSocket endpoint (P7) — streams the same OpenAI delta chunks as SSE over a socket.
/// The request JSON arrives as the first inbound Text frame (the upgrade is a GET, so
/// there's no body); each generated piece is sent as a Text frame carrying the same
/// [`sse_chunk`] payload, and the stream ends with a Close frame.
/// ponytail: request JSON = first text frame; no query-param / sub-protocol handshake.
async fn ws_upgrade(ws: axum::extract::ws::WebSocketUpgrade, State(app): State<AppState>) -> Response {
    // Reject an oversized frame at the protocol layer too (belt-and-suspenders with the
    // `start_request` body cap), so a huge first frame never fully buffers.
    ws.max_message_size(MAX_BODY_BYTES)
        .on_upgrade(move |socket| ws_conn(socket, app))
}

/// How long to wait for the client's first (request) frame before dropping the connection —
/// bounds an idle/slow-loris upgrade that would otherwise pin a task + socket forever (the
/// slot-based admission cap doesn't cover this pre-request window).
const WS_FIRST_FRAME_TIMEOUT: Duration = Duration::from_secs(30);

async fn ws_conn(mut socket: axum::extract::ws::WebSocket, app: AppState) {
    use axum::extract::ws::Message;
    // First inbound Text frame is the OpenAI request body — bounded by an ABSOLUTE deadline
    // (not a per-recv timeout, which a client could reset forever by dribbling a ping every
    // <30s) so a stalled/slow-loris upgrade can't hold the task/socket. A cap on non-Text
    // frames before the request bounds a ping flood within the window too.
    let deadline = tokio::time::Instant::now() + WS_FIRST_FRAME_TIMEOUT;
    let mut noise_budget = 64u32; // ping/pong/binary frames tolerated before the request frame
    let body = loop {
        match tokio::time::timeout_at(deadline, socket.recv()).await {
            Ok(Some(Ok(Message::Text(t)))) => break t.into_bytes(),
            Ok(Some(Ok(Message::Close(_)))) | Ok(None) => return, // client hung up before requesting
            Ok(Some(Ok(_))) => {
                // ping/pong/binary → keep waiting for text, but bounded.
                noise_budget = match noise_budget.checked_sub(1) {
                    Some(n) => n,
                    None => return, // too many non-request frames → drop
                };
            }
            Ok(Some(Err(_))) | Err(_) => return, // socket error or first-frame deadline
        }
    };
    let mut rx = match start_request(&app, &body) {
        Ok(rx) => rx,
        Err(e) => {
            // Already upgraded → surface the rejection as a text frame + close (not an HTTP
            // status). `sse_chunk`'s escaper (sonic_rs) also produces a JSON-safe string.
            let err = sonic_rs::to_string(e.msg).unwrap_or_else(|_| "\"error\"".into());
            let _ = socket.send(Message::Text(format!(r#"{{"error":{err}}}"#))).await;
            let _ = socket.close().await;
            return;
        }
    };
    while let Some(egress) = rx.next().await {
        match egress {
            Egress::Chunk(text) => {
                if socket.send(Message::Text(sse_chunk(&text))).await.is_err() {
                    return; // client disconnected; the slot still recycles on Finish via egress_loop
                }
            }
            Egress::Done => break,
        }
    }
    let _ = socket.close().await;
}

/// Minimal OpenAI-shaped streaming delta so `curl` output reads as chat completions.
fn sse_chunk(text: &str) -> String {
    let content = sonic_rs::to_string(text).unwrap_or_else(|_| "\"\"".into());
    format!(
        r#"{{"object":"chat.completion.chunk","choices":[{{"index":0,"delta":{{"content":{content}}}}}]}}"#
    )
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

    /// P7 WebSocket: the request JSON arrives as the first Text frame; each generated
    /// piece comes back as a `chat.completion.chunk` Text frame (identical OpenAI delta to
    /// SSE), then the server closes. Reconstructing the reply proves WS rides the same
    /// egress path as SSE. Real ephemeral-port bind + a `tokio-tungstenite` client, mirroring
    /// `serve_until_returns_on_shutdown`'s real-socket pattern.
    #[test]
    fn ws_streams_content_frames_and_closes() {
        use futures_util::SinkExt; // ws.send(); `next` is fully-qualified (tokio_stream also in scope)
        use tokio_tungstenite::tungstenite::Message as TMessage;

        let expected = std::str::from_utf8(CANNED_REPLY).expect("canned reply is valid utf8");
        let cfg = test_cfg(2);
        let slab = Arc::new(Slab::new(2, 128, 512, 256));
        let Pipeline { ingress, egress, core1, core2 } = pipeline::spawn(&cfg, Arc::clone(&slab));
        let state = AppState::new(slab, ingress);

        // Multi-thread runtime (dev-dependency feature): the server accept loop, the egress
        // drain, and the client must run concurrently — a single cooperative thread stalls.
        let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().unwrap();
        rt.block_on(async {
            let egress_task = tokio::spawn(egress_loop(
                Arc::clone(&state.slab),
                Arc::clone(&state.conns),
                Arc::clone(&state.free),
                Arc::clone(&state.cursor),
                egress,
            ));
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap();
            let app = build_app(state.clone());
            let serve_task = tokio::spawn(async move {
                let _ = axum::serve(listener, app).await;
            });

            let (mut ws, _resp) = tokio_tungstenite::connect_async(format!("ws://{addr}/v1/ws"))
                .await
                .expect("ws connect");
            // First text frame = the OpenAI request. max_tokens 5 → 5 ASCII bytes ("Hello").
            let body = r#"{"model":"m","messages":[{"role":"u","content":"hi"}],"max_tokens":5}"#;
            ws.send(TMessage::Text(body.into())).await.unwrap();

            let mut content = String::new();
            let mut frames = 0usize;
            let mut closed = false;
            while let Some(msg) = futures_util::StreamExt::next(&mut ws).await {
                match msg.expect("ws frame") {
                    TMessage::Text(t) => {
                        frames += 1;
                        let v: sonic_rs::Value = sonic_rs::from_str(&t).expect("delta json");
                        if let Some(c) = v["choices"][0]["delta"]["content"].as_str() {
                            content.push_str(c);
                        }
                    }
                    TMessage::Close(_) => {
                        closed = true;
                        break;
                    }
                    _ => {}
                }
            }
            assert_eq!(frames, 5, "one content frame per token (max_tokens=5)");
            assert_eq!(content, &expected[..5], "WS reconstructs the reply prefix (\"Hello\")");
            assert!(closed, "server closed the WS stream on finish");

            // Abort the server + egress tasks so they drop their `AppState` clone (which holds
            // the R1 ingress producer). Otherwise the clone below keeps R1 alive and the
            // core1/core2 join hangs forever.
            serve_task.abort();
            egress_task.abort();
        });

        drop(state); // last ingress holder → cascades R1→R4 Shutdown so the workers exit
        core1.join().expect("core1 join");
        core2.join().expect("core2 join");
    }

    /// P7 WebSocket trust boundary: an invalid first frame (bad JSON) is rejected *after* the
    /// upgrade as a `{"error":..}` Text frame + Close — not an HTTP status — and claims no
    /// slot, so a subsequent valid connection on the same (capacity-1) server still streams.
    #[test]
    fn ws_rejects_invalid_request_then_serves_a_valid_one() {
        use futures_util::SinkExt;
        use tokio_tungstenite::tungstenite::Message as TMessage;

        let expected = std::str::from_utf8(CANNED_REPLY).expect("canned reply is valid utf8");
        let cfg = test_cfg(1); // single slot → a leaked slot would wedge the 2nd request
        let slab = Arc::new(Slab::new(1, 128, 512, 256));
        let Pipeline { ingress, egress, core1, core2 } = pipeline::spawn(&cfg, Arc::clone(&slab));
        let state = AppState::new(slab, ingress);

        let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().unwrap();
        rt.block_on(async {
            let egress_task = tokio::spawn(egress_loop(
                Arc::clone(&state.slab),
                Arc::clone(&state.conns),
                Arc::clone(&state.free),
                Arc::clone(&state.cursor),
                egress,
            ));
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap();
            let app = build_app(state.clone());
            let serve_task = tokio::spawn(async move {
                let _ = axum::serve(listener, app).await;
            });
            let url = format!("ws://{addr}/v1/ws");

            // 1) Invalid first frame → one Text frame mentioning "error", then a Close.
            let (mut bad, _) = tokio_tungstenite::connect_async(&url).await.expect("connect");
            bad.send(TMessage::Text("not json".into())).await.unwrap();
            let mut saw_error = false;
            let mut closed = false;
            while let Some(msg) = futures_util::StreamExt::next(&mut bad).await {
                match msg.expect("frame") {
                    TMessage::Text(t) => {
                        assert!(t.contains("error"), "rejection frame carries an error: {t}");
                        saw_error = true;
                    }
                    TMessage::Close(_) => { closed = true; break; }
                    _ => {}
                }
            }
            assert!(saw_error && closed, "invalid request → error frame + close");

            // 2) A valid connection on the same capacity-1 server still streams (no slot leak).
            let (mut good, _) = tokio_tungstenite::connect_async(&url).await.expect("connect 2");
            good.send(TMessage::Text(
                r#"{"model":"m","messages":[{"role":"u","content":"hi"}],"max_tokens":5}"#.into(),
            )).await.unwrap();
            let mut content = String::new();
            while let Some(msg) = futures_util::StreamExt::next(&mut good).await {
                match msg.expect("frame") {
                    TMessage::Text(t) => {
                        let v: sonic_rs::Value = sonic_rs::from_str(&t).expect("delta json");
                        if let Some(c) = v["choices"][0]["delta"]["content"].as_str() {
                            content.push_str(c);
                        }
                    }
                    TMessage::Close(_) => break,
                    _ => {}
                }
            }
            assert_eq!(content, &expected[..5], "valid request streams after the rejection (slot not leaked)");

            serve_task.abort();
            egress_task.abort();
        });

        drop(state);
        core1.join().expect("core1 join");
        core2.join().expect("core2 join");
    }
}
