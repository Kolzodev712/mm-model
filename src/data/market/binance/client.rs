//! Live Binance partial-depth client.
//!
//! WebSocket frames, ping/pong, and reconnect waits stay in this file.
//! Callers see [`crate::data::market::source::BookSource`].

use std::io::ErrorKind;
use std::net::TcpStream;
use std::thread;
use std::time::Duration;

use tungstenite::stream::MaybeTlsStream;
use tungstenite::{connect, Message, WebSocket};

use crate::data::market::binance::parser::{
    parse_partial_depth, BinanceBookConfig, BINANCE_EXCHANGE,
};
use crate::data::market::source::BookSource;
use crate::domain::book::Instrument;
use crate::domain::book::ParsedDepth;
use crate::runtime::clock::Clock;
use crate::runtime::reconnect::ReconnectPolicy;

/// Failure from the Binance transport, after reconnect attempts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClientError {
    /// Symbol or endpoint was rejected before connect.
    Config,
    /// The socket could not be opened.
    Connect,
    /// A text frame was not a partial-depth snapshot.
    Malformed,
    /// The reconnect budget was used up.
    ReconnectExhausted,
    /// The socket had no frame before the read timeout. The caller should poll again.
    Timeout,
}

impl std::fmt::Display for ClientError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::Config => "binance book configuration was rejected",
            Self::Connect => "binance websocket connection failed",
            Self::Malformed => "binance book message was malformed",
            Self::ReconnectExhausted => "binance reconnect attempts were exhausted",
            Self::Timeout => "binance read timed out",
        };
        f.write_str(message)
    }
}

impl std::error::Error for ClientError {}

/// One partial-depth connection with bounded reconnect.
pub struct BinanceBookClient<C> {
    url: String,
    clock: C,
    socket: WebSocket<MaybeTlsStream<TcpStream>>,
    depth: ParsedDepth,
    instrument: Instrument,
    policy: ReconnectPolicy,
    reconnects: u64,
    consecutive_failures: u32,
    consecutive_malformed: u32,
    silent_timeouts: u32,
}

impl<C: Clock> BinanceBookClient<C> {
    /// Connects once. The stream URL is built here and stored.
    pub fn connect(
        config: &BinanceBookConfig,
        clock: C,
        policy: ReconnectPolicy,
    ) -> Result<Self, ClientError> {
        let url = config.stream_url();
        let symbol = config.symbol().to_ascii_uppercase();
        let instrument =
            Instrument::new(BINANCE_EXCHANGE, &symbol).map_err(|_| ClientError::Config)?;
        let mut socket = open_socket(&url)?;
        apply_timeouts(&mut socket);
        Ok(Self {
            url,
            clock,
            socket,
            depth: ParsedDepth::empty(),
            instrument,
            policy,
            reconnects: 0,
            consecutive_failures: 0,
            consecutive_malformed: 0,
            silent_timeouts: 0,
        })
    }

    #[must_use]
    pub fn reconnects(&self) -> u64 {
        self.reconnects
    }

    fn read_frame(&mut self) -> Result<crate::domain::book::BookRecord, ClientError> {
        let mut ignored = 0u32;
        loop {
            match self.socket.read() {
                Ok(Message::Text(payload)) => {
                    self.silent_timeouts = 0;
                    let local_ns = self.clock.unix_nanos().map_err(|_| ClientError::Connect)?;
                    if parse_partial_depth(payload.as_str(), &mut self.depth).is_err() {
                        return Err(ClientError::Malformed);
                    }
                    let record = crate::domain::book::BookRecord::from_depth(
                        self.instrument,
                        local_ns,
                        &self.depth,
                    )
                    .map_err(|_| ClientError::Malformed)?;
                    return Ok(record);
                }
                Ok(Message::Ping(payload)) => {
                    self.silent_timeouts = 0;
                    self.socket
                        .send(Message::Pong(payload))
                        .map_err(|_| ClientError::Connect)?;
                }
                Ok(Message::Close(_)) => return Err(ClientError::Connect),
                Ok(Message::Binary(_) | Message::Frame(_) | Message::Pong(_)) => {
                    self.silent_timeouts = 0;
                    ignored += 1;
                    if ignored > 32 {
                        return Err(ClientError::Connect);
                    }
                }
                Err(err) if is_timeout(&err) => return Err(ClientError::Timeout),
                Err(_) => return Err(ClientError::Connect),
            }
        }
    }

    fn reconnect_until_open(&mut self) -> Result<(), ClientError> {
        loop {
            self.consecutive_failures = self.consecutive_failures.saturating_add(1);
            self.reconnects = self.reconnects.saturating_add(1);
            let wait = self
                .policy
                .backoff(self.consecutive_failures)
                .ok_or(ClientError::ReconnectExhausted)?;
            thread::sleep(wait);
            match open_socket(&self.url) {
                Ok(mut socket) => {
                    apply_timeouts(&mut socket);
                    self.socket = socket;
                    return Ok(());
                }
                Err(ClientError::Connect) => continue,
                Err(err) => return Err(err),
            }
        }
    }
}

impl<C: Clock> BookSource for BinanceBookClient<C> {
    type Error = ClientError;

    fn next_book(&mut self) -> Result<Option<crate::domain::book::BookRecord>, Self::Error> {
        loop {
            match self.read_frame() {
                Ok(book) => {
                    self.consecutive_failures = 0;
                    self.consecutive_malformed = 0;
                    return Ok(Some(book));
                }
                Err(ClientError::Timeout) => {
                    self.silent_timeouts = self.silent_timeouts.saturating_add(1);
                    if self.silent_timeouts < SILENT_TIMEOUTS_BEFORE_RECONNECT {
                        return Err(ClientError::Timeout);
                    }
                    self.silent_timeouts = 0;
                    self.reconnect_until_open()?;
                }
                Err(ClientError::Malformed) => {
                    self.consecutive_malformed = self.consecutive_malformed.saturating_add(1);
                    if self.consecutive_malformed >= 8 {
                        return Err(ClientError::Malformed);
                    }
                }
                Err(ClientError::Connect) => self.reconnect_until_open()?,
                Err(err) => return Err(err),
            }
        }
    }

    fn is_idle(&self, err: &Self::Error) -> bool {
        matches!(err, ClientError::Timeout)
    }
}

/// Two-second reads. Fifteen in a row is about 30 seconds with no frame,
/// which is past a normal Binance ping, so the socket is treated as dead.
const SILENT_TIMEOUTS_BEFORE_RECONNECT: u32 = 15;

fn is_timeout(err: &tungstenite::Error) -> bool {
    match err {
        tungstenite::Error::Io(io) => {
            matches!(io.kind(), ErrorKind::TimedOut | ErrorKind::WouldBlock)
        }
        _ => false,
    }
}

fn apply_timeouts(socket: &mut WebSocket<MaybeTlsStream<TcpStream>>) {
    let timeout = Some(Duration::from_secs(2));
    let tcp = match socket.get_mut() {
        MaybeTlsStream::Plain(tcp) => tcp,
        MaybeTlsStream::Rustls(tls) => tls.get_mut(),
        _ => return,
    };
    let _ = tcp.set_read_timeout(timeout);
    let _ = tcp.set_write_timeout(timeout);
}

fn open_socket(url: &str) -> Result<WebSocket<MaybeTlsStream<TcpStream>>, ClientError> {
    // rustls 0.23 does not install a process-wide provider unless asked.
    // Doing this once at connect time keeps the per-message path unchanged.
    let _ = rustls::crypto::ring::default_provider().install_default();
    connect(url)
        .map(|(socket, _response)| socket)
        .map_err(|_| ClientError::Connect)
}
