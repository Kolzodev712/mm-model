//! Live Binance partial-depth client.
//!
//! WebSocket frames, ping/pong, and reconnect waits stay in this file.
//! Callers see [`crate::source::BookItem`].

use std::net::TcpStream;
use std::thread;

use tungstenite::stream::MaybeTlsStream;
use tungstenite::{connect, Message, WebSocket};

use crate::clock::Clock;
use crate::marketdata::{
    parse_partial_depth, BinanceBookConfig, Instrument, ParsedDepth, BINANCE_EXCHANGE,
};
use crate::reconnect::ReconnectPolicy;
use crate::source::BookSource;

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
}

impl std::fmt::Display for ClientError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::Config => "binance book configuration was rejected",
            Self::Connect => "binance websocket connection failed",
            Self::Malformed => "binance book message was malformed",
            Self::ReconnectExhausted => "binance reconnect attempts were exhausted",
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
        let socket = open_socket(&url)?;
        Ok(Self {
            url,
            clock,
            socket,
            depth: ParsedDepth::empty(),
            instrument,
            policy,
            reconnects: 0,
        })
    }

    #[must_use]
    pub fn reconnects(&self) -> u64 {
        self.reconnects
    }

    fn read_frame(&mut self) -> Result<crate::marketdata::BookRecord, ClientError> {
        loop {
            match self.socket.read() {
                Ok(Message::Text(payload)) => {
                    let local_ns = self.clock.unix_nanos().map_err(|_| ClientError::Connect)?;
                    if parse_partial_depth(payload.as_str(), &mut self.depth).is_err() {
                        return Err(ClientError::Malformed);
                    }
                    let record = crate::marketdata::BookRecord::from_depth(
                        self.instrument,
                        local_ns,
                        &self.depth,
                    )
                    .map_err(|_| ClientError::Malformed)?;
                    return Ok(record);
                }
                Ok(Message::Ping(payload)) => {
                    self.socket
                        .send(Message::Pong(payload))
                        .map_err(|_| ClientError::Connect)?;
                }
                Ok(Message::Close(_)) => return Err(ClientError::Connect),
                Ok(Message::Binary(_) | Message::Frame(_) | Message::Pong(_)) => {}
                Err(_) => return Err(ClientError::Connect),
            }
        }
    }

    fn reconnect(&mut self) -> Result<(), ClientError> {
        self.reconnects += 1;
        let attempt = u32::try_from(self.reconnects).unwrap_or(u32::MAX);
        let wait = self
            .policy
            .backoff(attempt)
            .ok_or(ClientError::ReconnectExhausted)?;
        thread::sleep(wait);
        self.socket = open_socket(&self.url)?;
        Ok(())
    }
}

impl<C: Clock> BookSource for BinanceBookClient<C> {
    type Error = ClientError;

    fn next_book(&mut self) -> Result<Option<crate::marketdata::BookRecord>, Self::Error> {
        loop {
            match self.read_frame() {
                Ok(book) => return Ok(Some(book)),
                Err(ClientError::Malformed) => return Err(ClientError::Malformed),
                Err(ClientError::Connect) => self.reconnect()?,
                Err(err) => return Err(err),
            }
        }
    }
}

fn open_socket(url: &str) -> Result<WebSocket<MaybeTlsStream<TcpStream>>, ClientError> {
    // rustls 0.23 does not install a process-wide provider unless asked.
    // Doing this once at connect time keeps the per-message path unchanged.
    let _ = rustls::crypto::ring::default_provider().install_default();
    connect(url)
        .map(|(socket, _response)| socket)
        .map_err(|_| ClientError::Connect)
}
