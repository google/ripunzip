// Copyright 2022 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use http::{StatusCode, Version};
use regex::Regex;

/// A response for use with `httptest` which is aware of HTTP ranges.
pub(crate) struct RangeAwareResponse(u16, Vec<u8>);

impl RangeAwareResponse {
    pub(crate) fn new(status_code: u16, body: Vec<u8>) -> Self {
        Self(status_code, body)
    }
}

impl httptest::responders::Responder for RangeAwareResponse {
    fn respond<'a>(
        &mut self,
        req: &'a httptest::http::Request<httptest::bytes::Bytes>,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = httptest::http::Response<hyper::Body>> + Send + 'a>,
    > {
        async fn _respond(resp: http::Response<hyper::Body>) -> http::Response<hyper::Body> {
            resp
        }
        let mut builder = http::Response::builder();
        builder = builder
            .status(StatusCode::from_u16(self.0).unwrap())
            .version(Version::HTTP_2)
            .header("Accept-Ranges", "bytes");
        let (body, content_length) = if let Some(range) = req.headers().get(http::header::RANGE) {
            let range_regex = Regex::new(r"bytes=(\d+)-(\d+)").unwrap();
            if let Some(captures) = range_regex.captures(range.to_str().unwrap()) {
                let from = captures
                    .get(1)
                    .and_then(|s| s.as_str().parse::<usize>().ok())
                    .unwrap();
                let to = captures
                    .get(2)
                    .and_then(|s| s.as_str().parse::<usize>().ok())
                    .unwrap();
                (self.1[from..to].to_vec(), to - from)
            } else {
                (self.1.clone(), self.1.len())
            }
        } else {
            (self.1.clone(), self.1.len())
        };
        let resp = builder
            .header("Content-Length", format!("{}", content_length))
            .body(body.into())
            .unwrap();

        Box::pin(_respond(resp))
    }
}
