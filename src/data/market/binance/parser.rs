//! Binance spot partial-book payload parser.
//!
//! Stream: `btcusdt@depth10@100ms`
//! URL: `wss://stream.binance.com:9443/ws/btcusdt@depth10@100ms`
//!
//! The official partial-depth payload is a top-N snapshot:
//!
//! ```text
//! {"lastUpdateId":160,"bids":[["price","qty"], ...],"asks":[["price","qty"], ...]}
//! ```
//!
//! It has no exchange event time. A numeric `"E"` field is accepted when a
//! payload includes one and is stored as unix milliseconds. Diff-depth messages
//! (`"e":"depthUpdate"`, `"b"`, `"a"`) are not snapshots and are rejected.
//!
//! Parsing writes into a reused [`ParsedDepth`]. It does not build a JSON tree
//! and does not allocate.

use std::fmt;

use crate::domain::book::{Level, ParsedDepth, BOOK_DEPTH};

/// Binance spot exchange name stored on each record.
pub const BINANCE_EXCHANGE: &str = "binance";
/// Instrument recorded by the reference recorder.
pub const BTCUSDT: &str = "BTCUSDT";
/// Example public endpoint. Not a model assumption.
pub const BINANCE_SPOT_WS: &str = "wss://stream.binance.com:9443";

/// Example stream kept for documentation. Clients build their own URL.
pub const PARTIAL_DEPTH_URL: &str = "wss://stream.binance.com:9443/ws/btcusdt@depth10@100ms";

/// Connection settings for one partial-depth stream.
///
/// The URL is derived once from these fields. Symbol is not hardcoded inside
/// the model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinanceBookConfig {
    base_endpoint: String,
    symbol: String,
    depth: u16,
    update_ms: u64,
}

impl BinanceBookConfig {
    /// Spot partial book. Depth must be 5, 10, or 20. Update interval must be 100 or 1000 ms.
    pub fn new(
        base_endpoint: impl Into<String>,
        symbol: impl Into<String>,
        depth: u16,
        update_ms: u64,
    ) -> Result<Self, ParseError> {
        let base_endpoint = base_endpoint.into();
        let symbol = symbol.into();
        if base_endpoint.is_empty()
            || !(base_endpoint.starts_with("wss://") || base_endpoint.starts_with("ws://"))
        {
            return Err(ParseError::Malformed);
        }
        if symbol.is_empty() || !symbol.bytes().all(|byte| byte.is_ascii_alphanumeric()) {
            return Err(ParseError::Malformed);
        }
        if !matches!(depth, 5 | 10 | 20) || !matches!(update_ms, 100 | 1000) {
            return Err(ParseError::Malformed);
        }
        Ok(Self {
            base_endpoint: base_endpoint.trim_end_matches('/').to_string(),
            symbol,
            depth,
            update_ms,
        })
    }

    /// Default BTCUSDT top-10 100 ms example. Callers can choose another symbol.
    pub fn btcusdt_example() -> Self {
        Self::new(BINANCE_SPOT_WS, BTCUSDT, 10, 100).expect("example config is valid")
    }

    #[must_use]
    pub fn symbol(&self) -> &str {
        &self.symbol
    }

    #[must_use]
    pub fn depth(&self) -> u16 {
        self.depth
    }

    /// Stream URL. Allocates. Call this at startup, not per book update.
    #[must_use]
    pub fn stream_url(&self) -> String {
        let symbol = self.symbol.to_ascii_lowercase();
        if self.update_ms == 1000 {
            format!("{}/ws/{symbol}@depth{}", self.base_endpoint, self.depth)
        } else {
            format!(
                "{}/ws/{symbol}@depth{}@{}ms",
                self.base_endpoint, self.depth, self.update_ms
            )
        }
    }
}

/// The payload was not a usable partial-depth snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParseError {
    /// JSON shape, numbers, or prices were not usable.
    Malformed,
    /// A side was not strictly best-first.
    Unordered,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::Malformed => "partial depth payload is malformed",
            Self::Unordered => "partial depth levels are not strictly best-first",
        };
        f.write_str(message)
    }
}

impl std::error::Error for ParseError {}

/// Parses one partial-depth payload into `out`.
///
/// On success, index 0 of each side is the best level and counts are in `1..=10`.
/// Extra levels beyond 10 are checked for order and then dropped. On failure
/// `out` is left empty, so a partial book is not kept.
pub fn parse_partial_depth(payload: &str, out: &mut ParsedDepth) -> Result<(), ParseError> {
    *out = ParsedDepth::empty();
    if let Err(err) = parse_into(payload, out) {
        *out = ParsedDepth::empty();
        return Err(err);
    }
    Ok(())
}

fn parse_into(payload: &str, out: &mut ParsedDepth) -> Result<(), ParseError> {
    let mut scan = Scan {
        bytes: payload.as_bytes(),
        index: 0,
    };
    scan.expect(b'{')?;
    let mut saw_id = false;
    let mut saw_bids = false;
    let mut saw_asks = false;
    let mut saw_event = false;
    scan.skip_ws();
    if scan.peek() == Some(b'}') {
        return Err(ParseError::Malformed);
    }
    loop {
        let key = scan.parse_string()?;
        scan.expect(b':')?;
        match key {
            "lastUpdateId" => {
                if saw_id {
                    return Err(ParseError::Malformed);
                }
                out.last_update_id = scan.parse_u64()?;
                saw_id = true;
            }
            "E" => {
                if saw_event {
                    return Err(ParseError::Malformed);
                }
                out.event_time_ms = Some(scan.parse_u64()?);
                saw_event = true;
            }
            "bids" => {
                if saw_bids {
                    return Err(ParseError::Malformed);
                }
                parse_side(&mut scan, &mut out.bids, &mut out.bid_count, true)?;
                saw_bids = true;
            }
            "asks" => {
                if saw_asks {
                    return Err(ParseError::Malformed);
                }
                parse_side(&mut scan, &mut out.asks, &mut out.ask_count, false)?;
                saw_asks = true;
            }
            _ => scan.skip_value(0)?,
        }
        scan.skip_ws();
        match scan.bump()? {
            b',' => continue,
            b'}' => break,
            _ => return Err(ParseError::Malformed),
        }
    }
    scan.skip_ws();
    if scan.index != scan.bytes.len() || !saw_id || !saw_bids || !saw_asks {
        return Err(ParseError::Malformed);
    }
    if out.bid_count == 0 || out.ask_count == 0 {
        return Err(ParseError::Malformed);
    }
    Ok(())
}

fn parse_side(
    scan: &mut Scan<'_>,
    levels: &mut [Level; BOOK_DEPTH],
    count: &mut u8,
    bids: bool,
) -> Result<(), ParseError> {
    scan.expect(b'[')?;
    scan.skip_ws();
    if scan.peek() == Some(b']') {
        scan.index += 1;
        return Ok(());
    }
    let mut previous: Option<f64> = None;
    loop {
        scan.expect(b'[')?;
        let price: f64 = scan
            .parse_string()?
            .parse()
            .map_err(|_| ParseError::Malformed)?;
        scan.expect(b',')?;
        let quantity: f64 = scan
            .parse_string()?
            .parse()
            .map_err(|_| ParseError::Malformed)?;
        scan.expect(b']')?;
        if !price.is_finite() || price <= 0.0 || !quantity.is_finite() || quantity <= 0.0 {
            return Err(ParseError::Malformed);
        }
        if let Some(prior) = previous {
            let ordered = if bids { price < prior } else { price > prior };
            if !ordered {
                return Err(ParseError::Unordered);
            }
        }
        previous = Some(price);
        let index = *count as usize;
        if index < BOOK_DEPTH {
            levels[index] = Level { price, quantity };
            *count += 1;
        }
        scan.skip_ws();
        match scan.bump()? {
            b',' => continue,
            b']' => return Ok(()),
            _ => return Err(ParseError::Malformed),
        }
    }
}

struct Scan<'a> {
    bytes: &'a [u8],
    index: usize,
}

impl<'a> Scan<'a> {
    fn skip_ws(&mut self) {
        while self.peek().is_some_and(|byte| byte.is_ascii_whitespace()) {
            self.index += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.index).copied()
    }

    fn bump(&mut self) -> Result<u8, ParseError> {
        let byte = self.peek().ok_or(ParseError::Malformed)?;
        self.index += 1;
        Ok(byte)
    }

    fn expect(&mut self, expected: u8) -> Result<(), ParseError> {
        self.skip_ws();
        if self.bump()? != expected {
            return Err(ParseError::Malformed);
        }
        Ok(())
    }

    fn parse_string(&mut self) -> Result<&'a str, ParseError> {
        self.skip_ws();
        if self.bump()? != b'"' {
            return Err(ParseError::Malformed);
        }
        let start = self.index;
        while let Some(byte) = self.peek() {
            if byte == b'"' {
                let text = std::str::from_utf8(&self.bytes[start..self.index])
                    .map_err(|_| ParseError::Malformed)?;
                self.index += 1;
                if !text.is_ascii() {
                    return Err(ParseError::Malformed);
                }
                return Ok(text);
            }
            if byte == b'\\' || !byte.is_ascii() {
                return Err(ParseError::Malformed);
            }
            self.index += 1;
        }
        Err(ParseError::Malformed)
    }

    fn parse_u64(&mut self) -> Result<u64, ParseError> {
        self.skip_ws();
        let start = self.index;
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            self.index += 1;
        }
        if start == self.index {
            return Err(ParseError::Malformed);
        }
        let text = std::str::from_utf8(&self.bytes[start..self.index])
            .map_err(|_| ParseError::Malformed)?;
        text.parse().map_err(|_| ParseError::Malformed)
    }

    fn skip_value(&mut self, depth: u8) -> Result<(), ParseError> {
        if depth > 8 {
            return Err(ParseError::Malformed);
        }
        self.skip_ws();
        match self.peek() {
            Some(b'{') => {
                self.index += 1;
                self.skip_ws();
                if self.peek() == Some(b'}') {
                    self.index += 1;
                    return Ok(());
                }
                loop {
                    self.parse_string()?;
                    self.expect(b':')?;
                    self.skip_value(depth + 1)?;
                    self.skip_ws();
                    match self.bump()? {
                        b',' => continue,
                        b'}' => return Ok(()),
                        _ => return Err(ParseError::Malformed),
                    }
                }
            }
            Some(b'[') => {
                self.index += 1;
                self.skip_ws();
                if self.peek() == Some(b']') {
                    self.index += 1;
                    return Ok(());
                }
                loop {
                    self.skip_value(depth + 1)?;
                    self.skip_ws();
                    match self.bump()? {
                        b',' => continue,
                        b']' => return Ok(()),
                        _ => return Err(ParseError::Malformed),
                    }
                }
            }
            Some(b'"') => {
                self.parse_string()?;
                Ok(())
            }
            Some(b't') => self.consume(b"true"),
            Some(b'f') => self.consume(b"false"),
            Some(b'n') => self.consume(b"null"),
            Some(b'-' | b'0'..=b'9') => self.skip_number(),
            _ => Err(ParseError::Malformed),
        }
    }

    fn consume(&mut self, literal: &[u8]) -> Result<(), ParseError> {
        for expected in literal {
            if self.bump()? != *expected {
                return Err(ParseError::Malformed);
            }
        }
        Ok(())
    }

    fn skip_number(&mut self) -> Result<(), ParseError> {
        if self.peek() == Some(b'-') {
            self.index += 1;
        }
        let start = self.index;
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            self.index += 1;
        }
        if self.peek() == Some(b'.') {
            self.index += 1;
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.index += 1;
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.index += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.index += 1;
            }
            let exp = self.index;
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.index += 1;
            }
            if self.index == exp {
                return Err(ParseError::Malformed);
            }
        }
        if self.index == start {
            return Err(ParseError::Malformed);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::book::{BookRecord, Instrument};

    #[test]
    fn parses_a_partial_depth_snapshot_in_best_first_order() {
        let payload = r#"{
            "lastUpdateId": 160,
            "bids": [["100.50", "1.25"], ["100.40", "2"]],
            "asks": [["100.70", "0.50"], ["100.80", "3.00"]]
        }"#;
        let mut depth = ParsedDepth::empty();
        parse_partial_depth(payload, &mut depth).unwrap();
        assert_eq!(depth.last_update_id, 160);
        assert_eq!(depth.event_time_ms, None);
        assert_eq!(depth.bid_count, 2);
        assert_eq!(depth.ask_count, 2);
        assert_eq!(depth.bids[0].price, 100.50);
        assert_eq!(depth.bids[0].quantity, 1.25);
        assert_eq!(depth.bids[1].price, 100.40);
        assert_eq!(depth.asks[0].price, 100.70);
        assert_eq!(depth.asks[0].quantity, 0.50);
        assert!(depth.bids[0].price > depth.bids[1].price);
        assert!(depth.asks[0].price < depth.asks[1].price);

        let record = BookRecord::from_depth(
            Instrument::new(BINANCE_EXCHANGE, BTCUSDT).unwrap(),
            1_700_000_000_000_000_000,
            &depth,
        )
        .unwrap();
        assert_eq!(record.stamp().local_receive_ns, 1_700_000_000_000_000_000);
        assert_eq!(record.stamp().exchange_event_ns, None);
        assert_eq!(record.stamp().update_id, Some(160));
        assert_eq!(record.instrument().exchange(), "binance");
        assert_eq!(record.instrument().symbol(), "BTCUSDT");
    }

    #[test]
    fn preserves_exchange_event_time_when_present_and_keeps_only_ten_levels() {
        let mut bids = String::from("[");
        let mut asks = String::from("[");
        for index in 0..12 {
            if index > 0 {
                bids.push(',');
                asks.push(',');
            }
            let bid = 100.0 - index as f64;
            let ask = 101.0 + index as f64;
            bids.push_str(&format!("[\"{bid}\",\"1\"]"));
            asks.push_str(&format!("[\"{ask}\",\"1\"]"));
        }
        bids.push(']');
        asks.push(']');
        let payload =
            format!(r#"{{"lastUpdateId":99,"E":1672515782136,"bids":{bids},"asks":{asks}}}"#);
        let mut depth = ParsedDepth::empty();
        parse_partial_depth(&payload, &mut depth).unwrap();
        assert_eq!(depth.event_time_ms, Some(1_672_515_782_136));
        assert_eq!(depth.bid_count, 10);
        assert_eq!(depth.ask_count, 10);
        assert_eq!(depth.bids[0].price, 100.0);
        assert_eq!(depth.bids[9].price, 91.0);
        assert_eq!(depth.asks[0].price, 101.0);
        assert_eq!(depth.asks[9].price, 110.0);

        let record = BookRecord::from_depth(
            Instrument::new(BINANCE_EXCHANGE, BTCUSDT).unwrap(),
            7,
            &depth,
        )
        .unwrap();
        assert_eq!(
            record.stamp().exchange_event_ns,
            Some(1_672_515_782_136_000_000)
        );
    }

    #[test]
    fn malformed_and_unordered_payloads_fail() {
        let mut depth = ParsedDepth::empty();
        assert_eq!(
            parse_partial_depth("{", &mut depth),
            Err(ParseError::Malformed)
        );
        assert_eq!(
            parse_partial_depth(r#"{"bids":[["1","1"]],"asks":[["2","1"]]}"#, &mut depth),
            Err(ParseError::Malformed)
        );
        assert_eq!(
            parse_partial_depth(
                r#"{"lastUpdateId":1,"bids":[],"asks":[["2","1"]]}"#,
                &mut depth
            ),
            Err(ParseError::Malformed)
        );
        assert_eq!(
            parse_partial_depth(
                r#"{"lastUpdateId":1,"bids":[["1","1"],["2","1"]],"asks":[["3","1"]]}"#,
                &mut depth
            ),
            Err(ParseError::Unordered)
        );
        assert_eq!(depth.bid_count, 0);
    }

    #[test]
    fn config_builds_the_partial_depth_url_once() {
        let config = BinanceBookConfig::new(BINANCE_SPOT_WS, "BTCUSDT", 10, 100).unwrap();
        assert_eq!(config.stream_url(), PARTIAL_DEPTH_URL);
        let slow = BinanceBookConfig::new(BINANCE_SPOT_WS, "ethusdt", 5, 1000).unwrap();
        assert_eq!(
            slow.stream_url(),
            "wss://stream.binance.com:9443/ws/ethusdt@depth5"
        );
        assert!(BinanceBookConfig::new(BINANCE_SPOT_WS, "BTCUSDT", 7, 100).is_err());
    }
}
