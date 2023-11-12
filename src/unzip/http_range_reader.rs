// Copyright 2022 Google LLC

// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::{cmp::min, io::Read};

use reqwest::blocking::{Client, Response};
use thiserror::Error;

/// Errors that may be returned by a [`RangeFetcher`].
#[derive(Debug, Error)]
pub(crate) enum Error {
    #[error("Initial HTTP HEAD command failed")]
    HttpHead(#[source] reqwest::Error),
    #[error("HTTP server did not specify a Content-Length header")]
    NoContentLength,
    #[error("HTTP resource was zero length")]
    EmptyContentLength,
    #[error("HTTP GET command failed")]
    HttpGet(#[source] reqwest::Error),
    #[error("Reading while fast-forwarding to desired location failed")]
    FastForward(#[source] std::io::Error),
}

/// An object which can fetch different ranges of a URI, using the HTTP
/// 'accept-range' header where supported. This fetcher works with HTTP servers
/// that either support or do not support 'accept-range' but can only be used
/// on HTTP resources which (a) report a `Content-Length`, and (b) do not change
/// between requests.
pub(crate) struct RangeFetcher {
    uri: String,
    accept_ranges: bool,
    content_length: u64,
    client: Client,
}

impl RangeFetcher {
    /// Create a new range fetcher for a given resource.
    pub(crate) fn new(uri: String) -> Result<Self, Error> {
        let client = reqwest::blocking::Client::new();
        let response = client.head(&uri).send().map_err(Error::HttpHead)?;
        let content_length = content_length_via_headers(&response).ok_or(Error::NoContentLength)?;
        if content_length == 0 {
            return Err(Error::EmptyContentLength);
        }
        let accept_ranges = response
            .headers()
            .contains_key(reqwest::header::ACCEPT_RANGES);
        Ok(Self {
            uri,
            accept_ranges,
            content_length,
            client,
        })
    }

    /// Return the total length of the resource.
    pub(crate) fn len(&self) -> u64 {
        self.content_length
    }

    /// Whether this resource supports fetch of specific ranges.
    pub(crate) fn accepts_ranges(&self) -> bool {
        self.accept_ranges
    }

    /// Return a [`Read`] for this resource starting from the given offset.
    /// If the resource supports HTTP ranges, this will start reading from
    /// the server at that point; otherwise, it will read from the outset
    /// of the resource but discard bytes before that point. (Clearly that
    /// can be expensive if you only care about a few bytes later in a
    /// resource.)
    pub(crate) fn fetch_range(&self, offset: u64) -> Result<Response, Error> {
        log::debug!("Fetch range 0x{:x}", offset);
        let mut builder = self.client.get(&self.uri);
        if self.accept_ranges {
            let range_header = format!("bytes={}-{}", offset, self.len());
            builder = builder.header(reqwest::header::RANGE, range_header);
        }
        let mut response = builder.send().map_err(Error::HttpGet)?;
        if !self.accept_ranges && offset > 0 {
            // Read and discard data prior to 'offset'
            let mut to_read = offset as usize;
            let mut throwaway = [0u8; 4096];
            while to_read > 0 {
                let bytes_read = response
                    .read(&mut throwaway[0..min(4096usize, to_read)])
                    .map_err(Error::FastForward)?;
                to_read -= bytes_read;
            }
        }
        Ok(response)
    }
}

/// Determine the `Content-Length` header. `reqwest` says it does this, but
/// doesn't: https://github.com/seanmonstar/reqwest/issues/1136
fn content_length_via_headers(response: &Response) -> Option<u64> {
    response
        .headers()
        .get(reqwest::header::CONTENT_LENGTH)
        .and_then(|hv| hv.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok())
}

#[cfg(test)]
mod tests {
    use httptest::{matchers::*, responders::*, Expectation, Server};
    use ripunzip_test_utils::*;
    use std::io::Read;
    use test_log::test;

    use super::RangeFetcher;

    fn do_test(accept_ranges: bool) {
        let server = Server::run();
        let body = "0123456789";
        server.expect(if accept_ranges {
            Expectation::matching(request::method_path("HEAD", "/foo"))
                .times(..)
                .respond_with(RangeAwareResponse::new(
                    200,
                    RangeAwareResponseType::Body {
                        body: hyper::body::Bytes::from(body),
                        expected_range: None,
                    },
                ))
        } else {
            Expectation::matching(request::method_path("HEAD", "/foo"))
                .times(..)
                .respond_with(
                    status_code(200).insert_header("Content-Length", format!("{}", body.len())),
                )
        });

        let range_fetcher = RangeFetcher::new(server.url("/foo").to_string()).unwrap();

        // Test reading the whole thing
        server.expect(if accept_ranges {
            Expectation::matching(any())
                .times(..)
                .respond_with(RangeAwareResponse::new(
                    206,
                    RangeAwareResponseType::Body {
                        body: hyper::body::Bytes::from(body),
                        expected_range: None,
                    },
                ))
        } else {
            Expectation::matching(any())
                .times(..)
                .respond_with(status_code(200).body(body))
        });
        assert_eq!(accept_ranges, range_fetcher.accepts_ranges());
        let mut resp = range_fetcher.fetch_range(0u64).unwrap();
        let mut throwaway = [0u8; 10];
        resp.read_exact(&mut throwaway).unwrap();
        assert_eq!(std::str::from_utf8(&throwaway).unwrap(), "0123456789");

        // Test read only a range
        let mut resp = range_fetcher.fetch_range(4u64).unwrap();
        let mut throwaway = [0u8; 6];
        resp.read_exact(&mut throwaway).unwrap();
        assert_eq!(std::str::from_utf8(&throwaway).unwrap(), "456789");
    }

    #[test]
    fn test_with_accept_range() {
        do_test(true);
    }

    #[test]
    fn test_without_accept_range() {
        do_test(false);
    }
}
